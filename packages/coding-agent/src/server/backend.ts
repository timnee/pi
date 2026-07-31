import { isAbsolute, join, resolve } from "node:path";
import type { Session } from "@earendil-works/pi-agent-core";
import { NodeExecutionEnv } from "@earendil-works/pi-agent-core/node";
import type { Api, Model } from "@earendil-works/pi-ai";
import type { ModelMetadata, ModelRef, SessionSummary, ThinkingLevel } from "@earendil-works/pi-protocol";
import {
	type CreateSessionOptions,
	PiServerError,
	type PiSessionBackend,
	type PiSessionRuntime,
} from "@earendil-works/pi-server";
import { getAgentDir } from "../config.ts";
import { ModelRuntime } from "../core/model-runtime.ts";
import { SettingsManager } from "../core/settings-manager.ts";
import { toPiServerError } from "./errors.ts";
import { CodingAgentModelCatalog } from "./model-catalog.ts";
import { type SessionLease, SessionLockManager } from "./session-lock.ts";
import { CodingAgentSessionPersistence } from "./session-persistence.ts";
import { CodingAgentSessionRuntime } from "./session-runtime.ts";

const DEFAULT_SESSION_ROOT_NAME = "server-sessions";

export interface CodingAgentServerBackendOptions {
	agentDir?: string;
	sessionRoot?: string;
	defaultCwd?: string;
	modelRuntime?: ModelRuntime;
	settingsManager?: SettingsManager;
}

export class CodingAgentServerBackend implements PiSessionBackend {
	private readonly modelRuntime: ModelRuntime;
	private readonly settingsManager: SettingsManager;
	private readonly defaultCwd: string;
	private readonly models: CodingAgentModelCatalog;
	private readonly persistence: CodingAgentSessionPersistence;
	private readonly locks: SessionLockManager;

	private constructor(options: {
		modelRuntime: ModelRuntime;
		settingsManager: SettingsManager;
		sessionRoot: string;
		defaultCwd: string;
	}) {
		this.modelRuntime = options.modelRuntime;
		this.settingsManager = options.settingsManager;
		this.defaultCwd = options.defaultCwd;
		this.models = new CodingAgentModelCatalog(this.modelRuntime, this.settingsManager);
		this.persistence = new CodingAgentSessionPersistence(
			new NodeExecutionEnv({ cwd: this.defaultCwd }),
			options.sessionRoot,
		);
		this.locks = new SessionLockManager(options.sessionRoot);
	}

	static async create(options: CodingAgentServerBackendOptions = {}): Promise<CodingAgentServerBackend> {
		const agentDir = resolve(options.agentDir ?? getAgentDir());
		const defaultCwd = resolve(options.defaultCwd ?? process.cwd());
		const settingsSource =
			options.settingsManager ?? SettingsManager.create(defaultCwd, agentDir, { projectTrusted: false });
		const settingsManager = SettingsManager.inMemory(settingsSource.getGlobalSettings(), { projectTrusted: false });
		const modelRuntime =
			options.modelRuntime ??
			(await ModelRuntime.create({
				authPath: join(agentDir, "auth.json"),
				modelsPath: join(agentDir, "models.json"),
				allowModelNetwork: false,
			}));
		await modelRuntime.getAvailable();
		const sessionRoot = resolve(options.sessionRoot ?? join(agentDir, DEFAULT_SESSION_ROOT_NAME));
		const backend = new CodingAgentServerBackend({ modelRuntime, settingsManager, sessionRoot, defaultCwd });
		await validateCwd(defaultCwd);
		await backend.locks.initialize();
		return backend;
	}

	setDefaultSessionOptions(options: { model?: ModelRef; thinkingLevel?: ThinkingLevel }): void {
		this.models.setDefaults(options);
	}

	listModels(): Promise<ModelMetadata[]> {
		return this.models.list();
	}

	async listSessions(): Promise<SessionSummary[]> {
		const stored = await this.persistence.list();
		return Promise.all(
			stored.map(async ({ metadata, state, createdAt }) => {
				try {
					if (!state.model) throw new Error("stored session has no model");
					if (state.invalidThinkingLevel !== undefined) {
						throw new Error(`stored session has invalid thinking level: ${state.invalidThinkingLevel}`);
					}
					return {
						id: metadata.id,
						name: state.name,
						cwd: metadata.cwd,
						createdAt,
						updatedAt: state.updatedAt,
						phase: "idle" as const,
						model: state.model,
						thinkingLevel: state.thinkingLevel ?? "off",
						attached: false,
						locked: await this.locks.isLocked(metadata.id),
					};
				} catch (error) {
					throw new Error(`Failed to read coding-agent session ${metadata.id}`, { cause: error });
				}
			}),
		);
	}

	async createSession(options: CreateSessionOptions): Promise<PiSessionRuntime> {
		const cwd = options.cwd ?? this.defaultCwd;
		await validateCwd(cwd);
		const model = await this.models.resolve(options.model);
		const thinkingLevel = this.models.resolveThinkingLevel(model, options.thinkingLevel);
		const lease = await this.locks.acquire(options.id);
		let session: Session | undefined;
		try {
			if (await this.persistence.find(options.id)) {
				throw new PiServerError("invalid_request", `Session already exists: ${options.id}`);
			}
			session = await this.persistence.create({ cwd, id: options.id });
			await session.appendModelChange(model.provider, model.id);
			await session.appendThinkingLevelChange(thinkingLevel);
			if (options.name !== undefined) await session.appendSessionName(options.name);
			return await this.createRuntime(session, model, thinkingLevel, lease);
		} catch (error) {
			const cleanupErrors: unknown[] = [];
			if (session) {
				try {
					await this.persistence.delete(session);
				} catch (cleanupError) {
					cleanupErrors.push(cleanupError);
				}
			}
			try {
				await lease.release();
			} catch (cleanupError) {
				cleanupErrors.push(cleanupError);
			}
			if (cleanupErrors.length > 0) {
				throw new AggregateError([error, ...cleanupErrors], `Failed to create and clean up session ${options.id}`);
			}
			throw toPiServerError(error);
		}
	}

	async openSession(sessionId: string): Promise<PiSessionRuntime> {
		const metadata = await this.persistence.find(sessionId);
		if (!metadata) throw new PiServerError("not_found", `Session was not found: ${sessionId}`);
		const lease = await this.locks.acquire(sessionId);
		try {
			const session = await this.persistence.open(metadata);
			const { state } = await this.persistence.inspect(session);
			if (!state.model) throw new PiServerError("invalid_request", `Session ${sessionId} has no saved model`);
			if (state.invalidThinkingLevel !== undefined) {
				throw new PiServerError(
					"invalid_request",
					`Session ${sessionId} has invalid thinking level: ${state.invalidThinkingLevel}`,
				);
			}
			const model = await this.models.resolve(state.model);
			const thinkingLevel = this.models.recoverThinkingLevel(model, state.thinkingLevel);
			if (state.thinkingLevel !== thinkingLevel) await session.appendThinkingLevelChange(thinkingLevel);
			return await this.createRuntime(session, model, thinkingLevel, lease);
		} catch (error) {
			await lease.release();
			throw toPiServerError(error);
		}
	}

	private async createRuntime(
		session: Session,
		model: Model<Api>,
		thinkingLevel: ThinkingLevel,
		lease: SessionLease,
	): Promise<CodingAgentSessionRuntime> {
		const metadata = await session.getMetadata();
		if (!("cwd" in metadata) || typeof metadata.cwd !== "string") {
			throw new PiServerError("invalid_request", "Session metadata is missing cwd");
		}
		return CodingAgentSessionRuntime.create({
			session,
			persistence: this.persistence,
			modelRuntime: this.modelRuntime,
			settingsManager: this.settingsManager,
			model,
			thinkingLevel,
			lease,
			cwd: metadata.cwd,
		});
	}
}

async function validateCwd(cwd: string): Promise<void> {
	if (!isAbsolute(cwd)) throw new PiServerError("invalid_request", `Session cwd must be absolute: ${cwd}`);
	const env = new NodeExecutionEnv({ cwd });
	try {
		const info = await env.fileInfo(cwd);
		if (!info.ok || info.value.kind !== "directory") {
			const suffix = info.ok ? "is not a directory" : info.error.message;
			throw new PiServerError("invalid_request", `Invalid session cwd ${cwd}: ${suffix}`);
		}
	} finally {
		await env.cleanup();
	}
}

export { toPiServerError } from "./errors.ts";
