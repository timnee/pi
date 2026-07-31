import {
	AgentHarness,
	type AgentHarnessStreamOptions,
	type AgentHarnessTool,
	createBashTool,
	createEditTool,
	createReadTool,
	createWriteTool,
	type ExecutionToolContext,
	type Session,
} from "@earendil-works/pi-agent-core";
import type { NodeExecutionEnv } from "@earendil-works/pi-agent-core/node";
import type { Api, Model } from "@earendil-works/pi-ai";
import type { ThinkingLevel } from "@earendil-works/pi-protocol";
import type { ModelRuntime } from "../core/model-runtime.ts";
import { mergeProviderAttributionHeaders } from "../core/provider-attribution.ts";
import type { SettingsManager } from "../core/settings-manager.ts";
import { buildSystemPrompt } from "../core/system-prompt.ts";

export interface CreateServerHarnessOptions {
	session: Session;
	modelRuntime: ModelRuntime;
	settingsManager: SettingsManager;
	model: Model<Api>;
	thinkingLevel: ThinkingLevel;
	env: NodeExecutionEnv;
	streamOptions: AgentHarnessStreamOptions;
	assertUsable(): void;
}

export function createServerHarness(options: CreateServerHarnessOptions): AgentHarness<ExecutionToolContext> {
	let harness: AgentHarness<ExecutionToolContext>;
	const readTool = createReadTool();
	const bashTool = createBashTool({
		commandPrefix: options.settingsManager.getShellCommandPrefix(),
		promptGuidelines: ["Inspect PI_* environment variables for current model and session details."],
		prepare: async (execution) => {
			options.assertUsable();
			const metadata = await options.session.getMetadata();
			execution.env.PI_SESSION_ID = metadata.id;
			if ("path" in metadata && typeof metadata.path === "string") execution.env.PI_SESSION_FILE = metadata.path;
			execution.env.PI_PROVIDER = harness.getModel().provider;
			execution.env.PI_MODEL = harness.getModel().id;
			execution.env.PI_REASONING_LEVEL = harness.getThinkingLevel();
		},
	});
	const tools: AgentHarnessTool<ExecutionToolContext>[] = [readTool, bashTool, createEditTool(), createWriteTool()];
	const toolNames = tools.map((tool) => tool.name);
	const toolSnippets = Object.fromEntries(tools.map((tool) => [tool.name, tool.promptSnippet ?? tool.description]));
	const promptGuidelines = [
		...(bashTool.promptGuidelines ?? []),
		...tools.filter((tool) => tool !== bashTool).flatMap((tool) => tool.promptGuidelines ?? []),
	];
	harness = new AgentHarness<ExecutionToolContext>({
		session: options.session,
		models: options.modelRuntime,
		model: options.model,
		thinkingLevel: options.thinkingLevel,
		tools,
		activeToolNames: toolNames,
		toolContext: { env: options.env },
		resources: {},
		steeringMode: options.settingsManager.getSteeringMode(),
		followUpMode: options.settingsManager.getFollowUpMode(),
		streamOptions: options.streamOptions,
		systemPrompt: async () => {
			options.assertUsable();
			const metadata = await options.session.getMetadata();
			const cwd = "cwd" in metadata && typeof metadata.cwd === "string" ? metadata.cwd : options.env.cwd;
			return buildSystemPrompt({
				cwd,
				selectedTools: toolNames,
				toolSnippets,
				promptGuidelines,
				contextFiles: [],
				skills: [],
			});
		},
	});
	harness.on("before_provider_request", (event) => {
		const merged = mergeProviderAttributionHeaders(event.model, options.settingsManager, event.sessionId);
		const headers = merged
			? Object.fromEntries(Object.entries(merged).filter((entry): entry is [string, string] => entry[1] !== null))
			: undefined;
		return { streamOptions: { headers } };
	});
	return harness;
}
