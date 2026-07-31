import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { PiServerError } from "@earendil-works/pi-server";
import lockfile from "proper-lockfile";

const STALE_MS = 30_000;
const UPDATE_MS = 10_000;

export interface SessionLease {
	compromisedError(): Error | undefined;
	onCompromised(listener: (error: Error) => void): () => void;
	release(): Promise<void>;
}

function errorCode(error: unknown): string | undefined {
	return typeof error === "object" && error !== null && "code" in error
		? String((error as { code?: unknown }).code)
		: undefined;
}

export class SessionLockManager {
	private readonly root: string;

	constructor(sessionRoot: string) {
		this.root = join(sessionRoot, ".locks");
	}

	async initialize(): Promise<void> {
		await mkdir(this.root, { recursive: true, mode: 0o700 });
	}

	async isLocked(sessionId: string): Promise<boolean> {
		return lockfile.check(await this.ensureTarget(sessionId), { realpath: false, stale: STALE_MS });
	}

	async acquire(sessionId: string): Promise<SessionLease> {
		const target = await this.ensureTarget(sessionId);
		let compromised: Error | undefined;
		const listeners = new Set<(error: Error) => void>();
		let releaseLock: (() => Promise<void>) | undefined;
		try {
			releaseLock = await lockfile.lock(target, {
				realpath: false,
				stale: STALE_MS,
				update: UPDATE_MS,
				retries: 0,
				onCompromised: (error) => {
					compromised = error;
					for (const listener of listeners) listener(error);
				},
			});
		} catch (error) {
			if (errorCode(error) === "ELOCKED") {
				throw new PiServerError("session_locked", `Session is locked: ${sessionId}`);
			}
			throw error;
		}
		return this.createLease(
			() => releaseLock?.(),
			listeners,
			() => compromised,
		);
	}

	private async ensureTarget(sessionId: string): Promise<string> {
		await this.initialize();
		const digest = createHash("sha256").update(sessionId, "utf8").digest("hex");
		const target = join(this.root, digest);
		await writeFile(target, "", { flag: "a", mode: 0o600 });
		return target;
	}

	private createLease(
		releaseLock: () => Promise<void> | undefined,
		listeners: Set<(error: Error) => void>,
		getCompromised: () => Error | undefined,
	): SessionLease {
		let released = false;
		let releasing: Promise<void> | undefined;
		return {
			compromisedError: getCompromised,
			onCompromised: (listener) => {
				listeners.add(listener);
				const compromised = getCompromised();
				if (compromised) listener(compromised);
				return () => listeners.delete(listener);
			},
			release: async () => {
				if (released) return;
				if (getCompromised()) {
					released = true;
					listeners.clear();
					return;
				}
				if (releasing) return releasing;
				const current = (async () => {
					await releaseLock();
					released = true;
					listeners.clear();
				})();
				releasing = current;
				try {
					await current;
				} finally {
					if (!released && releasing === current) releasing = undefined;
				}
			},
		};
	}
}
