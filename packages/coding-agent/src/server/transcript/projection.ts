import type { AgentMessage, SessionTreeEntry } from "@earendil-works/pi-agent-core";
import type { AssistantMessage, ToolCall, ToolResultMessage, UserMessage } from "@earendil-works/pi-ai";
import type { AssistantTranscriptItem, ToolTranscriptItem, TranscriptItem } from "@earendil-works/pi-protocol";
import {
	toProtocolAssistantMessage,
	toProtocolToolResultMessage,
	toProtocolUserMessage,
} from "@earendil-works/pi-server";

type FinishedAssistantTranscriptItem = Exclude<AssistantTranscriptItem, { status: "streaming" }>;
type FinishedToolTranscriptItem = Exclude<ToolTranscriptItem, { status: "running" }>;

function timestamp(value: number): number {
	if (!Number.isSafeInteger(value) || value < 0) throw new Error(`Invalid message timestamp: ${value}`);
	return value;
}

export function transcriptItemId(message: AgentMessage): string {
	if (message.role === "toolResult") return `tool-${message.toolCallId || timestamp(message.timestamp)}`;
	return `${message.role}-${timestamp(message.timestamp)}`;
}

export function projectAssistantMessage(
	message: AssistantMessage,
	streaming: true,
	id?: string,
): Extract<AssistantTranscriptItem, { status: "streaming" }>;
export function projectAssistantMessage(
	message: AssistantMessage,
	streaming?: false,
	id?: string,
): FinishedAssistantTranscriptItem;
export function projectAssistantMessage(
	message: AssistantMessage,
	streaming = false,
	id = transcriptItemId(message),
): AssistantTranscriptItem {
	const item = toProtocolAssistantMessage(streaming ? { ...message, stopReason: "pending" } : message, { id });
	if (streaming && item.status !== "streaming") throw new Error("Expected a streaming assistant transcript item");
	if (!streaming && item.status === "streaming") throw new Error("Expected a finished assistant transcript item");
	return item;
}

export function projectUserMessage(message: UserMessage, id = transcriptItemId(message)) {
	return toProtocolUserMessage(message, { id });
}

export function projectToolResult(
	message: ToolResultMessage,
	call?: ToolCall,
	id = transcriptItemId(message),
): FinishedToolTranscriptItem {
	const item = toProtocolToolResultMessage(message, {
		id,
		call: call ?? {
			type: "toolCall",
			id: message.toolCallId,
			name: message.toolName || "unknown",
			arguments: {},
		},
	});
	if (item.status === "running") throw new Error("Expected a finished tool transcript item");
	return item;
}

export function isAssistantMessage(message: AgentMessage): message is AssistantMessage {
	return message.role === "assistant";
}

export function isUserMessage(message: AgentMessage): message is UserMessage {
	return message.role === "user";
}

export function isToolResultMessage(message: AgentMessage): message is ToolResultMessage {
	return message.role === "toolResult";
}

export function projectBranchTranscript(entries: readonly SessionTreeEntry[]): TranscriptItem[] {
	const calls = new Map<string, ToolCall>();
	const transcript: TranscriptItem[] = [];
	for (const entry of entries) {
		if (entry.type !== "message") continue;
		const message = entry.message;
		if (isUserMessage(message)) {
			transcript.push(projectUserMessage(message, entry.id));
			continue;
		}
		if (isAssistantMessage(message)) {
			for (const part of message.content) {
				if (part.type === "toolCall") {
					calls.set(part.id, part);
				}
			}
			transcript.push(projectAssistantMessage(message, false, entry.id));
			continue;
		}
		if (isToolResultMessage(message)) {
			transcript.push(projectToolResult(message, calls.get(message.toolCallId), entry.id));
		}
	}
	return transcript;
}
