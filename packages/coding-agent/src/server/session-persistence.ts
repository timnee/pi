import {
	createJsonlSessionStore,
	createSessionRepository,
	type JsonlSessionCreateOptions,
	type JsonlSessionListOptions,
	type JsonlSessionMetadata,
	type Session,
	type SessionRepository,
	type SessionTreeEntry,
} from "@earendil-works/pi-agent-core";
import type { NodeExecutionEnv } from "@earendil-works/pi-agent-core/node";
import type { ThinkingLevel } from "@earendil-works/pi-protocol";
import { isAssistantMessage } from "./transcript/projection.ts";

export interface StoredSessionState {
	model: { provider: string; id: string } | undefined;
	thinkingLevel: ThinkingLevel | undefined;
	invalidThinkingLevel: string | undefined;
	name: string | undefined;
	updatedAt: number;
}

export interface StoredSession {
	metadata: JsonlSessionMetadata;
	state: StoredSessionState;
	createdAt: number;
}

const THINKING_LEVELS = new Set<ThinkingLevel>(["off", "minimal", "low", "medium", "high", "xhigh", "max"]);

export class CodingAgentSessionPersistence {
	private readonly repository: SessionRepository<
		JsonlSessionMetadata,
		JsonlSessionCreateOptions,
		JsonlSessionListOptions
	>;

	constructor(fs: NodeExecutionEnv, sessionsRoot: string) {
		this.repository = createSessionRepository({ store: createJsonlSessionStore({ fs, sessionsRoot }) });
	}

	async list(): Promise<StoredSession[]> {
		return Promise.all((await this.repository.list()).map((metadata) => this.load(metadata)));
	}

	async find(sessionId: string): Promise<JsonlSessionMetadata | undefined> {
		return (await this.repository.list()).find((metadata) => metadata.id === sessionId);
	}

	async create(options: { id: string; cwd: string }): Promise<Session> {
		return this.repository.create(options);
	}

	async open(metadata: JsonlSessionMetadata): Promise<Session> {
		return this.repository.open(metadata);
	}

	async delete(session: Session): Promise<void> {
		const metadata = await session.getMetadata();
		if (
			!("cwd" in metadata) ||
			typeof metadata.cwd !== "string" ||
			!("path" in metadata) ||
			typeof metadata.path !== "string"
		) {
			throw new Error(`Cannot delete session ${metadata.id}: JSONL metadata is incomplete`);
		}
		await this.repository.delete({ ...metadata, cwd: metadata.cwd, path: metadata.path });
	}

	async inspect(
		session: Session,
	): Promise<{ branch: SessionTreeEntry[]; state: StoredSessionState; createdAt: number }> {
		const metadata = await session.getMetadata();
		const branch = await getFullActiveBranch(session);
		const createdAt = parseCreatedAt(metadata.createdAt);
		return { branch, state: readStoredSessionState(branch, createdAt), createdAt };
	}

	private async load(metadata: JsonlSessionMetadata): Promise<StoredSession> {
		try {
			const session = await this.open(metadata);
			const { state, createdAt } = await this.inspect(session);
			return { metadata, state, createdAt };
		} catch (error) {
			throw new Error(`Failed to read coding-agent session ${metadata.id}`, { cause: error });
		}
	}
}

export async function getFullActiveBranch(session: Session): Promise<SessionTreeEntry[]> {
	const branch: SessionTreeEntry[] = [];
	const visited = new Set<string>();
	let id = await session.getLeafId();
	while (id !== null) {
		if (visited.has(id)) throw new Error(`Session branch contains a cycle at ${id}`);
		visited.add(id);
		const entry = await session.getEntry(id);
		if (!entry) throw new Error(`Session branch entry ${id} was not found`);
		branch.unshift(entry);
		id = entry.parentId;
	}
	return branch;
}

function readStoredSessionState(entries: readonly SessionTreeEntry[], createdAt: number): StoredSessionState {
	let model: StoredSessionState["model"];
	let thinkingLevel: ThinkingLevel | undefined;
	let invalidThinkingLevel: string | undefined;
	let name: string | undefined;
	let updatedAt = createdAt;
	for (const entry of entries) {
		const entryTime = Date.parse(entry.timestamp);
		if (!Number.isFinite(entryTime) || entryTime < 0) {
			throw new Error(`Invalid timestamp on session entry ${entry.id}: ${entry.timestamp}`);
		}
		updatedAt = Math.max(updatedAt, entryTime);
		if (entry.type === "model_change") model = { provider: entry.provider, id: entry.modelId };
		if (entry.type === "thinking_level_change") {
			if (THINKING_LEVELS.has(entry.thinkingLevel as ThinkingLevel)) {
				thinkingLevel = entry.thinkingLevel as ThinkingLevel;
				invalidThinkingLevel = undefined;
			} else {
				thinkingLevel = undefined;
				invalidThinkingLevel = entry.thinkingLevel;
			}
		}
		if (entry.type === "message" && isAssistantMessage(entry.message)) {
			model = { provider: entry.message.provider, id: entry.message.model };
		}
		if (entry.type === "session_info") name = entry.name?.trim() || undefined;
	}
	return { model, thinkingLevel, invalidThinkingLevel, name, updatedAt };
}

function parseCreatedAt(value: string): number {
	const parsed = Date.parse(value);
	if (!Number.isFinite(parsed) || parsed < 0) throw new Error(`Invalid session creation timestamp: ${value}`);
	return parsed;
}
