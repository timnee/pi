import type { AgentHarness, ExecutionToolContext, Session } from "@earendil-works/pi-agent-core";
import { NodeExecutionEnv } from "@earendil-works/pi-agent-core/node";
import { type Api, clampThinkingLevel, type Model } from "@earendil-works/pi-ai";
import type { ModelRef, SessionPhase, SessionSnapshot, ThinkingLevel } from "@earendil-works/pi-protocol";
import {
	PiServerError,
	type PiSessionRuntime,
	type PiSessionRuntimeEvent,
	type PromptInput,
	type SteerInput,
} from "@earendil-works/pi-server";
import type { ModelRuntime } from "../core/model-runtime.ts";
import type { SettingsManager } from "../core/settings-manager.ts";
import { createServerHarness } from "./create-harness.ts";
import { toPiServerError } from "./errors.ts";
import type { SessionLease } from "./session-lock.ts";
import type { CodingAgentSessionPersistence } from "./session-persistence.ts";
import { LiveTranscript } from "./transcript/live.ts";
import { projectBranchTranscript } from "./transcript/projection.ts";

export interface CreateCodingAgentSessionRuntimeOptions {
	session: Session;
	persistence: CodingAgentSessionPersistence;
	modelRuntime: ModelRuntime;
	settingsManager: SettingsManager;
	model: Model<Api>;
	thinkingLevel: ThinkingLevel;
	lease: SessionLease;
	cwd: string;
}

export class CodingAgentSessionRuntime implements PiSessionRuntime {
	private readonly session: Session;
	private readonly persistence: CodingAgentSessionPersistence;
	private readonly modelRuntime: ModelRuntime;
	private readonly env: NodeExecutionEnv;
	private readonly lease: SessionLease;
	private readonly harness: AgentHarness<ExecutionToolContext>;
	private readonly liveTranscript = new LiveTranscript((phase) => {
		this.phase = phase;
	});
	private readonly unsubscribeHarness: () => void;
	private readonly unsubscribeLockCompromise: () => void;
	private lockCompromise?: Error;
	private terminalError?: PiServerError;
	private mutationInFlight = false;
	private phase: SessionPhase = "idle";
	private disposed = false;
	private disposePromise?: Promise<void>;

	private constructor(options: CreateCodingAgentSessionRuntimeOptions, env: NodeExecutionEnv) {
		this.session = options.session;
		this.persistence = options.persistence;
		this.modelRuntime = options.modelRuntime;
		this.env = env;
		this.lease = options.lease;
		const retry = options.settingsManager.getProviderRetrySettings();
		const idleTimeoutMs = options.settingsManager.getHttpIdleTimeoutMs();
		this.harness = createServerHarness({
			session: this.session,
			modelRuntime: this.modelRuntime,
			settingsManager: options.settingsManager,
			model: options.model,
			thinkingLevel: options.thinkingLevel,
			env: this.env,
			streamOptions: {
				transport: options.settingsManager.getTransport(),
				timeoutMs: retry.timeoutMs ?? (idleTimeoutMs === 0 ? 2_147_483_647 : idleTimeoutMs),
				maxRetries: retry.maxRetries,
				maxRetryDelayMs: retry.maxRetryDelayMs,
			},
			assertUsable: () => this.assertUsable(),
		});
		this.unsubscribeHarness = this.harness.subscribe((event) => {
			try {
				this.liveTranscript.handle(event);
			} catch (error) {
				this.handleProjectionFailure(error);
			}
		});
		this.unsubscribeLockCompromise = this.lease.onCompromised((error) => this.handleLockCompromise(error));
	}

	static create(options: CreateCodingAgentSessionRuntimeOptions): CodingAgentSessionRuntime {
		const env = new NodeExecutionEnv({ cwd: options.cwd, shellPath: options.settingsManager.getShellPath() });
		return new CodingAgentSessionRuntime(options, env);
	}

	getPhase(): SessionPhase {
		return this.phase;
	}

	async snapshot(): Promise<SessionSnapshot> {
		this.assertUsable();
		const metadata = await this.session.getMetadata();
		if (!("cwd" in metadata) || typeof metadata.cwd !== "string") {
			throw new PiServerError("invalid_request", "Session metadata is missing cwd");
		}
		const { branch, state, createdAt } = await this.persistence.inspect(this.session);
		const queuedSteer = this.liveTranscript.queuedSteer;
		return {
			id: metadata.id,
			name: state.name,
			cwd: metadata.cwd,
			createdAt,
			updatedAt: state.updatedAt,
			phase: this.getPhase(),
			model: { provider: this.harness.getModel().provider, id: this.harness.getModel().id },
			thinkingLevel: this.harness.getThinkingLevel(),
			attached: false,
			locked: true,
			revision: this.liveTranscript.revision,
			transcript: this.liveTranscript.mergeTranscript(projectBranchTranscript(branch)),
			queuedSteer,
			queuedSteerCount: queuedSteer.length,
		};
	}

	async prompt(input: PromptInput): Promise<void> {
		this.assertIdle("prompt");
		this.phase = "turn";
		try {
			await this.harness.prompt(input.text);
			this.assertUsable();
		} catch (error) {
			this.assertUsable();
			throw toPiServerError(error);
		} finally {
			if (!this.lockCompromise) this.phase = "idle";
		}
	}

	async steer(input: SteerInput): Promise<void> {
		this.assertUsable();
		if (this.getPhase() !== "turn") throw new PiServerError("busy", "Session is not accepting steering input");
		try {
			await this.harness.steer(input.text);
			this.assertUsable();
		} catch (error) {
			this.assertUsable();
			throw toPiServerError(error);
		}
	}

	async abort(): Promise<void> {
		this.assertUsable();
		if (this.mutationInFlight) throw new PiServerError("busy", "Cannot abort while session is being updated");
		try {
			await this.harness.abort();
			this.assertUsable();
		} catch (error) {
			this.assertUsable();
			throw toPiServerError(error);
		}
	}

	async setModel(reference: ModelRef): Promise<void> {
		await this.runExclusiveMutation("set model", () => {
			const model = this.modelRuntime.getModel(reference.provider, reference.id);
			if (!model) throw new PiServerError("invalid_request", `Unknown model: ${reference.provider}/${reference.id}`);
			if (!this.modelRuntime.hasConfiguredAuth(model.provider)) {
				throw new PiServerError(
					"invalid_request",
					`Model is not authenticated: ${reference.provider}/${reference.id}`,
				);
			}
			return async () => {
				await this.harness.setModel(model);
				const current = this.harness.getThinkingLevel();
				const clamped = clampThinkingLevel(model, current);
				if (clamped !== current) await this.harness.setThinkingLevel(clamped);
			};
		});
	}

	async setThinking(thinkingLevel: ThinkingLevel): Promise<void> {
		await this.runExclusiveMutation("set thinking", () => {
			const model = this.harness.getModel();
			if (clampThinkingLevel(model, thinkingLevel) !== thinkingLevel) {
				throw new PiServerError(
					"invalid_request",
					`Thinking level ${thinkingLevel} is not supported by ${model.provider}/${model.id}`,
				);
			}
			return () => this.harness.setThinkingLevel(thinkingLevel);
		});
	}

	subscribe(listener: (event: PiSessionRuntimeEvent) => void): () => void {
		return this.liveTranscript.subscribe(listener);
	}

	async dispose(): Promise<void> {
		if (this.disposePromise) return this.disposePromise;
		this.disposed = true;
		const current = (async () => {
			try {
				this.unsubscribeHarness();
				this.unsubscribeLockCompromise();
				this.liveTranscript.clear();
				if (this.getPhase() !== "idle") await this.harness.abort().catch(() => {});
				await this.harness.waitForIdle().catch(() => {});
				await this.env.cleanup();
			} finally {
				await this.lease.release();
			}
		})();
		this.disposePromise = current;
		try {
			await current;
		} catch (error) {
			if (this.disposePromise === current) this.disposePromise = undefined;
			throw error;
		}
	}

	private assertUsable(): void {
		if (this.terminalError) throw this.terminalError;
		this.throwIfLockCompromised();
		if (this.disposed) throw new PiServerError("invalid_request", "Session runtime is disposed");
	}

	private throwIfLockCompromised(): void {
		const compromised = this.lockCompromise ?? this.lease.compromisedError();
		if (compromised) {
			throw new PiServerError("session_locked", `Session lock was compromised: ${compromised.message}`);
		}
	}

	private handleLockCompromise(error: Error): void {
		if (this.lockCompromise) return;
		this.lockCompromise = error;
		this.terminateHarness(new PiServerError("session_locked", `Session lock was compromised: ${error.message}`));
	}

	private handleProjectionFailure(error: unknown): void {
		if (this.terminalError) return;
		this.terminalError = new PiServerError("invalid_request", "Session protocol projection failed");
		this.terminalError.cause = error instanceof Error ? error : new Error(String(error));
		this.terminateHarness(this.terminalError);
	}

	private terminateHarness(error: PiServerError): void {
		this.unsubscribeHarness();
		this.harness.requestShutdown({ discardPendingWrites: true });
		this.liveTranscript.emit({ type: "error", error });
	}

	private assertIdle(operation: string): void {
		this.assertUsable();
		if (this.mutationInFlight || this.getPhase() !== "idle") {
			throw new PiServerError("busy", `Cannot ${operation} while session is busy`);
		}
	}

	private async runExclusiveMutation(operation: string, prepare: () => () => Promise<void>): Promise<void> {
		this.assertIdle(operation);
		const mutation = prepare();
		this.mutationInFlight = true;
		try {
			await mutation();
			this.assertUsable();
		} catch (error) {
			if (this.terminalError || this.lockCompromise || this.lease.compromisedError() || this.disposed) {
				this.assertUsable();
			}
			this.terminalError = new PiServerError("invalid_request", `Session failed to ${operation}`);
			this.terminalError.cause = toPiServerError(error);
			this.terminateHarness(this.terminalError);
			throw this.terminalError;
		} finally {
			this.mutationInFlight = false;
		}
	}
}
