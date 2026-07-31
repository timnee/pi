import { randomUUID } from "node:crypto";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type FauxResponseFactory, fauxAssistantMessage, fauxProvider } from "@earendil-works/pi-ai";
import { AuthStorage } from "../../src/core/auth-storage.ts";
import { ModelRuntime } from "../../src/core/model-runtime.ts";
import { SettingsManager } from "../../src/core/settings-manager.ts";
import { CodingAgentServerBackend } from "../../src/server/backend.ts";

export async function createServerBackendFixture() {
	const root = await mkdtemp(join(tmpdir(), "pi-experimental-backend-"));
	const cwd = join(root, "workspace");
	await mkdir(cwd);
	const faux = fauxProvider({
		provider: `faux-${randomUUID()}`,
		models: [
			{ id: "faux-reasoning", name: "Faux Reasoning", reasoning: true },
			{ id: "faux-plain", name: "Faux Plain", reasoning: false },
		],
		tokensPerSecond: 100,
		tokenSize: { min: 2, max: 3 },
	});
	const modelRuntime = await ModelRuntime.create({
		credentials: AuthStorage.inMemory(),
		modelsPath: null,
		allowModelNetwork: false,
	});
	modelRuntime.registerNativeProvider(faux.provider);
	await modelRuntime.refresh({ allowNetwork: false });
	const settingsManager = SettingsManager.inMemory({
		defaultProvider: faux.provider.id,
		defaultModel: "faux-reasoning",
		defaultThinkingLevel: "high",
		transport: "sse",
		retry: { provider: { timeoutMs: 5_000, maxRetries: 0, maxRetryDelayMs: 100 } },
	});
	const backend = await CodingAgentServerBackend.create({
		defaultCwd: cwd,
		sessionRoot: join(root, "sessions"),
		modelRuntime,
		settingsManager,
	});
	return { root, cwd, faux, modelRuntime, settingsManager, backend };
}

export async function removeServerBackendFixture(
	fixture: Awaited<ReturnType<typeof createServerBackendFixture>>,
): Promise<void> {
	await rm(fixture.root, { recursive: true, force: true, maxRetries: 5, retryDelay: 20 });
}

export function abortableResponse(): { response: FauxResponseFactory; started: Promise<void> } {
	let startedResolve: (() => void) | undefined;
	const started = new Promise<void>((resolve) => {
		startedResolve = resolve;
	});
	return {
		started,
		response: async (_context, options) => {
			startedResolve?.();
			const signal = options?.signal;
			if (!signal) throw new Error("Expected an abort signal");
			if (!signal.aborted) {
				await new Promise<void>((resolve) => signal.addEventListener("abort", () => resolve(), { once: true }));
			}
			return fauxAssistantMessage("aborted");
		},
	};
}
