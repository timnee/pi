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

const TOOL_NAMES = ["read", "bash", "edit", "write"] as const;
const TOOL_SNIPPETS: Record<(typeof TOOL_NAMES)[number], string> = {
	read: "Read file contents",
	bash: "Execute bash commands (ls, grep, find, etc.)",
	edit: "Make precise file edits with exact text replacement, including multiple disjoint edits in one call",
	write: "Create or overwrite files",
};
const TOOL_GUIDELINES = [
	"Inspect PI_* environment variables for current model and session details.",
	"Use read to examine files instead of cat or sed.",
	"Use edit for precise changes (edits[].oldText must match exactly)",
	"When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls",
	"Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.",
	"Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.",
	"Use write only for new files or complete rewrites.",
];

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
	const tools: AgentHarnessTool<ExecutionToolContext>[] = [
		createReadTool(),
		createBashTool({
			commandPrefix: options.settingsManager.getShellCommandPrefix(),
			prepare: async (execution) => {
				options.assertUsable();
				const metadata = await options.session.getMetadata();
				execution.env.PI_SESSION_ID = metadata.id;
				if ("path" in metadata && typeof metadata.path === "string") execution.env.PI_SESSION_FILE = metadata.path;
				execution.env.PI_PROVIDER = harness.getModel().provider;
				execution.env.PI_MODEL = harness.getModel().id;
				execution.env.PI_REASONING_LEVEL = harness.getThinkingLevel();
			},
		}),
		createEditTool(),
		createWriteTool(),
	];
	harness = new AgentHarness<ExecutionToolContext>({
		session: options.session,
		models: options.modelRuntime,
		model: options.model,
		thinkingLevel: options.thinkingLevel,
		tools,
		activeToolNames: [...TOOL_NAMES],
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
				selectedTools: [...TOOL_NAMES],
				toolSnippets: TOOL_SNIPPETS,
				promptGuidelines: TOOL_GUIDELINES,
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
