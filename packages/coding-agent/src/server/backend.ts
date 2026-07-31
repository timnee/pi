import { isAbsolute, join, resolve } from "node:path";
import {
	createJsonlSessionStore,
	createSessionRepository,
	type JsonlSessionCreateOptions,
	type JsonlSessionListOptions,
	type JsonlSessionMetadata,
	type Session,
	type SessionCreateOptions,
	type SessionMetadata,
	type SessionRepository,
} from "@earendil-works/pi-agent-core";
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
import { ServerModelResolver } from "./model-resolver.ts";
import { type SessionLease, SessionLockManager } from "./session-lock.ts";
import { CodingAgentSessionRuntime } from "./session-runtime.ts";
import { initializeStoredSession, inspectStoredSession } from "./session-state.ts";

const DEFAULT_SESSION_ROOT_NAME = "server-sessions";
const DEFAULT_LOCK_ROOT_NAME = "server-session-locks";

export interface CodingAgentServerBackendOptions {
	agentDir?: string;
	sessionRoot?: string;
	lockRoot?: string;
	defaultCwd?: string;
	modelRuntime?: ModelRuntime;
	settingsManager?: SettingsManager;
}

export interface CodingAgentServerRepositoryOptions<
	TMetadata extends SessionMetadata,
	TCreateOptions extends SessionCreateOptions,
	TListOptions,
> extends Omit<CodingAgentServerBackendOptions, "sessionRoot"> {
	sessionRepository: SessionRepository<TMetadata, TCreateOptions, TListOptions>;
	createSessionOptions(options: { id: string; cwd: string }): TCreateOptions;
}

export class CodingAgentServerBackend<
	TMetadata extends SessionMetadata = JsonlSessionMetadata,
	TCreateOptions extends SessionCreateOptions = JsonlSessionCreateOptions,
	TListOptions = JsonlSessionListOptions,
> implements PiSessionBackend
{
	private readonly modelRuntime: ModelRuntime;
	private readonly settingsManager: SettingsManager;
	private readonly defaultCwd: string;
	private readonly modelResolver: ServerModelResolver;
	private readonly sessions: SessionRepository<TMetadata, TCreateOptions, TListOptions>;
	private readonly createSessionOptions: (options: { id: string; cwd: string }) => TCreateOptions;
	private readonly locks: SessionLockManager;

	private constructor(options: {
		modelRuntime: ModelRuntime;
		settingsManager: SettingsManager;
		sessions: SessionRepository<TMetadata, TCreateOptions, TListOptions>;
		createSessionOptions(options: { id: string; cwd: string }): TCreateOptions;
		lockRoot: string;
		defaultCwd: string;
	}) {
		this.modelRuntime = options.modelRuntime;
		this.settingsManager = options.settingsManager;
		this.defaultCwd = options.defaultCwd;
		this.modelResolver = new ServerModelResolver(this.modelRuntime, this.settingsManager);
		this.sessions = options.sessions;
		this.createSessionOptions = options.createSessionOptions;
		this.locks = new SessionLockManager(options.lockRoot);
	}

	static create<TMetadata extends SessionMetadata, TCreateOptions extends SessionCreateOptions, TListOptions>(
		options: CodingAgentServerRepositoryOptions<TMetadata, TCreateOptions, TListOptions>,
	): Promise<CodingAgentServerBackend<TMetadata, TCreateOptions, TListOptions>>;
	static create(options?: CodingAgentServerBackendOptions): Promise<CodingAgentServerBackend>;
	static async create<TMetadata extends SessionMetadata, TCreateOptions extends SessionCreateOptions, TListOptions>(
		options:
			| CodingAgentServerBackendOptions
			| CodingAgentServerRepositoryOptions<TMetadata, TCreateOptions, TListOptions> = {},
	): Promise<CodingAgentServerBackend | CodingAgentServerBackend<TMetadata, TCreateOptions, TListOptions>> {
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
		await validateCwd(defaultCwd);

		if ("sessionRepository" in options) {
			const backend = new CodingAgentServerBackend<TMetadata, TCreateOptions, TListOptions>({
				modelRuntime,
				settingsManager,
				sessions: options.sessionRepository,
				createSessionOptions: options.createSessionOptions,
				lockRoot: resolve(options.lockRoot ?? join(agentDir, DEFAULT_LOCK_ROOT_NAME)),
				defaultCwd,
			});
			await backend.locks.initialize();
			return backend;
		}

		const sessionRoot = resolve(options.sessionRoot ?? join(agentDir, DEFAULT_SESSION_ROOT_NAME));
		const sessions = createSessionRepository({
			store: createJsonlSessionStore({ fs: new NodeExecutionEnv({ cwd: defaultCwd }), sessionsRoot: sessionRoot }),
		});
		const backend = new CodingAgentServerBackend({
			modelRuntime,
			settingsManager,
			sessions,
			createSessionOptions: ({ id, cwd }) => ({ id, cwd }),
			lockRoot: resolve(options.lockRoot ?? sessionRoot),
			defaultCwd,
		});
		await backend.locks.initialize();
		return backend;
	}

	setDefaultSessionOptions(options: { model?: ModelRef; thinkingLevel?: ThinkingLevel }): void {
		this.modelResolver.setDefaults(options);
	}

	listModels(): Promise<ModelMetadata[]> {
		return this.modelResolver.list();
	}

	async listSessions(): Promise<SessionSummary[]> {
		const stored = await this.sessions.list();
		return Promise.all(
			stored.map(async (metadata) => {
				try {
					const session = await this.sessions.open(metadata);
					const { state, createdAt } = await inspectStoredSession(session);
					if (!state.cwd) throw new Error("stored session has no cwd");
					if (!state.model) throw new Error("stored session has no model");
					if (state.invalidThinkingLevel !== undefined) {
						throw new Error(`stored session has invalid thinking level: ${state.invalidThinkingLevel}`);
					}
					return {
						id: metadata.id,
						name: state.name,
						cwd: state.cwd,
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
		const model = await this.modelResolver.resolve(options.model);
		const thinkingLevel = this.modelResolver.resolveThinkingLevel(model, options.thinkingLevel);
		const lease = await this.locks.acquire(options.id);
		let session: Session<TMetadata> | undefined;
		try {
			if ((await this.sessions.list()).some((metadata) => metadata.id === options.id)) {
				throw new PiServerError("invalid_request", `Session already exists: ${options.id}`);
			}
			session = await this.sessions.create(this.createSessionOptions({ cwd, id: options.id }));
			await initializeStoredSession(session, cwd);
			await session.appendModelChange(model.provider, model.id);
			await session.appendThinkingLevelChange(thinkingLevel);
			if (options.name !== undefined) await session.appendSessionName(options.name);
			return await this.createRuntime(session, model, thinkingLevel, lease, cwd);
		} catch (error) {
			const cleanupErrors: unknown[] = [];
			if (session) {
				try {
					await this.sessions.delete(await session.getMetadata());
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
		const metadata = (await this.sessions.list()).find((candidate) => candidate.id === sessionId);
		if (!metadata) throw new PiServerError("not_found", `Session was not found: ${sessionId}`);
		const lease = await this.locks.acquire(sessionId);
		try {
			const session = await this.sessions.open(metadata);
			const { state } = await inspectStoredSession(session);
			if (!state.cwd) throw new PiServerError("invalid_request", `Session ${sessionId} has no saved cwd`);
			if (!state.model) throw new PiServerError("invalid_request", `Session ${sessionId} has no saved model`);
			if (state.invalidThinkingLevel !== undefined) {
				throw new PiServerError(
					"invalid_request",
					`Session ${sessionId} has invalid thinking level: ${state.invalidThinkingLevel}`,
				);
			}
			const model = await this.modelResolver.resolve(state.model);
			const thinkingLevel = this.modelResolver.recoverThinkingLevel(model, state.thinkingLevel);
			if (state.thinkingLevel !== thinkingLevel) await session.appendThinkingLevelChange(thinkingLevel);
			return await this.createRuntime(session, model, thinkingLevel, lease, state.cwd);
		} catch (error) {
			await lease.release();
			throw toPiServerError(error);
		}
	}

	private async createRuntime(
		session: Session<TMetadata>,
		model: Model<Api>,
		thinkingLevel: ThinkingLevel,
		lease: SessionLease,
		cwd: string,
	): Promise<CodingAgentSessionRuntime> {
		return CodingAgentSessionRuntime.create({
			session,
			modelRuntime: this.modelRuntime,
			modelResolver: this.modelResolver,
			settingsManager: this.settingsManager,
			model,
			thinkingLevel,
			lease,
			cwd,
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
