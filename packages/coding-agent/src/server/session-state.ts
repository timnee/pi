import type { Session, SessionTreeEntry } from "@earendil-works/pi-agent-core";
import { type ThinkingLevel, ThinkingLevelSchema } from "@earendil-works/pi-protocol";
import { Check } from "typebox/value";
import { isAssistantMessage } from "./transcript/projection.ts";

const SERVER_SESSION_ENTRY_TYPE = "pi-server-session";

export interface ServerSessionState {
	cwd: string | undefined;
	model: { provider: string; id: string } | undefined;
	thinkingLevel: ThinkingLevel | undefined;
	invalidThinkingLevel: string | undefined;
	name: string | undefined;
	updatedAt: number;
}

export async function initializeServerSession(session: Session, cwd: string): Promise<void> {
	await session.appendCustomEntry(SERVER_SESSION_ENTRY_TYPE, { cwd });
}

export async function inspectServerSession(
	session: Session,
): Promise<{ branch: SessionTreeEntry[]; state: ServerSessionState; createdAt: number }> {
	const metadata = await session.getMetadata();
	const branch = await getFullActiveBranch(session);
	const createdAt = parseCreatedAt(metadata.createdAt);
	return { branch, state: readServerSessionState(branch, createdAt), createdAt };
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

function isThinkingLevel(value: string): value is ThinkingLevel {
	return Check(ThinkingLevelSchema, value);
}

function readServerSessionState(entries: readonly SessionTreeEntry[], createdAt: number): ServerSessionState {
	let cwd: string | undefined;
	let model: ServerSessionState["model"];
	let thinkingLevel: ThinkingLevel | undefined;
	let invalidThinkingLevel: string | undefined;
	let name: string | undefined;
	let updatedAt = createdAt;
	for (const entry of entries) {
		const entryTime = parseEntryTimestamp(entry.timestamp, entry.id);
		updatedAt = Math.max(updatedAt, entryTime);
		if (entry.type === "custom" && entry.customType === SERVER_SESSION_ENTRY_TYPE) {
			if (
				typeof entry.data !== "object" ||
				entry.data === null ||
				!("cwd" in entry.data) ||
				typeof entry.data.cwd !== "string" ||
				entry.data.cwd.length === 0
			) {
				throw new Error(`Invalid server session metadata on entry ${entry.id}`);
			}
			cwd = entry.data.cwd;
		}
		if (entry.type === "model_change") model = { provider: entry.provider, id: entry.modelId };
		if (entry.type === "thinking_level_change") {
			if (isThinkingLevel(entry.thinkingLevel)) {
				thinkingLevel = entry.thinkingLevel;
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
	return { cwd, model, thinkingLevel, invalidThinkingLevel, name, updatedAt };
}

function parseCreatedAt(value: string): number {
	const parsed = Date.parse(value);
	if (!Number.isFinite(parsed) || parsed < 0) throw new Error(`Invalid session creation timestamp: ${value}`);
	return parsed;
}

function parseEntryTimestamp(value: string, entryId: string): number {
	const parsed = Date.parse(value);
	if (!Number.isFinite(parsed) || parsed < 0) {
		throw new Error(`Invalid timestamp on session entry ${entryId}: ${value}`);
	}
	return parsed;
}
