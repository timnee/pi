import type { AgentHarnessEvent, AgentMessage } from "@earendil-works/pi-agent-core";
import type { ToolCall } from "@earendil-works/pi-ai";
import type { JsonValue, ToolTranscriptItem, TranscriptItem, TranscriptProgress } from "@earendil-works/pi-protocol";
import {
	type PiSessionRuntimeEvent,
	sanitizeProtocolDetails,
	toProtocolJsonValue,
	toProtocolUsage,
} from "@earendil-works/pi-server";
import {
	isAssistantMessage,
	isToolResultMessage,
	isUserMessage,
	projectAssistantMessage,
	projectToolResult,
	projectUserMessage,
	transcriptItemId,
} from "./projection.ts";

type RunningToolTranscriptItem = Extract<ToolTranscriptItem, { status: "running" }>;
type FinishedToolTranscriptItem = Exclude<ToolTranscriptItem, { status: "running" }>;

interface ProjectedToolResult {
	content: ToolTranscriptItem["content"];
	details?: JsonValue;
	usage?: ToolTranscriptItem["usage"];
}

function projectToolProgressResult(value: unknown): ProjectedToolResult {
	if (typeof value !== "object" || value === null) return { content: [] };
	const source = value as { content?: unknown; details?: unknown; usage?: unknown };
	const content: ToolTranscriptItem["content"] = [];
	if (Array.isArray(source.content)) {
		for (const part of source.content) {
			if (typeof part !== "object" || part === null || !("type" in part)) continue;
			const candidate = part as { type?: unknown; text?: unknown; data?: unknown; mimeType?: unknown };
			if (candidate.type === "text" && typeof candidate.text === "string") {
				content.push({ type: "text", text: candidate.text });
			} else if (
				candidate.type === "image" &&
				typeof candidate.data === "string" &&
				typeof candidate.mimeType === "string" &&
				candidate.mimeType
			) {
				content.push({ type: "image", data: candidate.data, mimeType: candidate.mimeType });
			}
		}
	}
	const details = sanitizeProtocolDetails(source.details);
	const usage = toProtocolUsage(source.usage as Parameters<typeof toProtocolUsage>[0]);
	return {
		content,
		...(details === undefined ? {} : { details }),
		...(usage === undefined ? {} : { usage }),
	};
}

export class LiveTranscript {
	private readonly onPhase: (phase: "idle" | "turn") => void;
	private readonly listeners = new Set<(event: PiSessionRuntimeEvent) => void>();
	private readonly liveItems = new Map<string, TranscriptItem>();
	private readonly liveOrder: string[] = [];
	private readonly toolInputs = new Map<string, { call: ToolCall; input: JsonValue; timestamp: number }>();
	private queuedSteerItems: ReturnType<typeof projectUserMessage>[] = [];
	revision = 0;

	constructor(onPhase: (phase: "idle" | "turn") => void) {
		this.onPhase = onPhase;
	}

	get queuedSteer(): ReturnType<typeof projectUserMessage>[] {
		return structuredClone(this.queuedSteerItems);
	}

	mergeTranscript(persisted: readonly TranscriptItem[]): TranscriptItem[] {
		const result = persisted.map((item) => this.liveItems.get(item.id) ?? item);
		const persistedIds = new Set(persisted.map((item) => item.id));
		for (const id of this.liveOrder) {
			if (persistedIds.has(id)) continue;
			const item = this.liveItems.get(id);
			if (item) result.push(item);
		}
		return result;
	}

	subscribe(listener: (event: PiSessionRuntimeEvent) => void): () => void {
		this.listeners.add(listener);
		return () => this.listeners.delete(listener);
	}

	clear(): void {
		this.listeners.clear();
	}

	emit(event: PiSessionRuntimeEvent): void {
		for (const listener of this.listeners) {
			try {
				listener(event);
			} catch {
				// A disconnected transport subscriber cannot affect the harness.
			}
		}
	}

	handle(event: AgentHarnessEvent): void {
		if (event.type === "agent_start") this.onPhase("turn");
		if (event.type === "agent_end" || event.type === "settled") this.onPhase("idle");
		if (event.type === "message_start") {
			this.handleMessageStart(event.message);
			return;
		}
		if (event.type === "message_update" && isAssistantMessage(event.message)) {
			const item = projectAssistantMessage(event.message, true);
			this.remember(item);
			const update = event.assistantMessageEvent;
			if (update.type === "text_delta" || update.type === "thinking_delta" || update.type === "toolcall_delta") {
				this.emitProgress({
					type: "assistant_delta",
					messageId: item.id,
					contentIndex: update.contentIndex,
					kind: update.type === "text_delta" ? "text" : update.type === "thinking_delta" ? "thinking" : "toolCall",
					delta: update.delta,
				});
			} else this.emitProgress({ type: "item_updated", item });
			return;
		}
		if (event.type === "message_end") {
			this.handleMessageEnd(event.message);
			return;
		}
		if (event.type === "tool_execution_start") {
			const input = toProtocolJsonValue(event.args);
			const startedAt = Date.now();
			this.toolInputs.set(event.toolCallId, {
				call: { type: "toolCall", id: event.toolCallId, name: event.toolName, arguments: event.args },
				input,
				timestamp: startedAt,
			});
			const item = this.createToolItem(
				event.toolCallId,
				event.toolName,
				input,
				{ content: [] },
				"running",
				startedAt,
			);
			this.remember(item);
			this.emitProgress({ type: "item_started", item });
			return;
		}
		if (event.type === "tool_execution_update") {
			const call = this.toolInputs.get(event.toolCallId) ?? {
				call: { type: "toolCall", id: event.toolCallId, name: event.toolName, arguments: event.args },
				input: toProtocolJsonValue(event.args),
				timestamp: Date.now(),
			};
			this.toolInputs.set(event.toolCallId, call);
			const item = this.createToolItem(
				event.toolCallId,
				event.toolName,
				call.input,
				projectToolProgressResult(event.partialResult),
				"running",
				call.timestamp,
			);
			this.remember(item);
			this.emitProgress({ type: "item_updated", item });
			return;
		}
		if (event.type === "tool_execution_end") {
			const call = this.toolInputs.get(event.toolCallId) ?? {
				call: { type: "toolCall", id: event.toolCallId, name: event.toolName, arguments: {} },
				input: null,
				timestamp: Date.now(),
			};
			const item = this.createToolItem(
				event.toolCallId,
				event.toolName,
				call.input,
				projectToolProgressResult(event.result),
				event.isError ? "error" : "complete",
				call.timestamp,
			);
			this.remember(item);
			this.emitProgress({ type: "item_finished", item });
			return;
		}
		if (event.type === "queue_update") {
			this.queuedSteerItems = event.steer
				.filter(isUserMessage)
				.map((message, index) => projectUserMessage(message, `queued-steer-${message.timestamp}-${index}`));
			this.emitChange();
			return;
		}
		if (
			[
				"agent_start",
				"agent_end",
				"model_update",
				"thinking_level_update",
				"abort",
				"settled",
				"save_point",
			].includes(event.type)
		)
			this.emitChange();
	}

	private handleMessageStart(message: AgentMessage): void {
		if (isAssistantMessage(message)) {
			const item = projectAssistantMessage(message, true);
			this.remember(item);
			this.emitChange({ type: "item_started", item });
		} else if (isUserMessage(message)) {
			const item = projectUserMessage(message);
			this.remember(item);
			this.emitChange({ type: "item_started", item });
		} else this.emitChange();
	}

	private handleMessageEnd(message: Parameters<typeof transcriptItemId>[0]): void {
		const id = transcriptItemId(message);
		if (isAssistantMessage(message)) {
			const item = projectAssistantMessage(message);
			this.remember(item);
			this.revision += 1;
			this.emit({ type: "progress", progress: { type: "item_finished", item } });
			this.forget(id);
			this.emit({ type: "snapshot" });
		} else if (isToolResultMessage(message)) {
			const call = this.toolInputs.get(message.toolCallId);
			const item = projectToolResult(message, call?.call);
			this.remember(item);
			this.forget(item.id);
			this.toolInputs.delete(message.toolCallId);
			this.emitChange();
		} else {
			this.forget(id);
			this.emitChange();
		}
	}

	private remember(item: TranscriptItem): void {
		if (!this.liveItems.has(item.id)) this.liveOrder.push(item.id);
		this.liveItems.set(item.id, item);
	}

	private forget(id: string): void {
		this.liveItems.delete(id);
		const index = this.liveOrder.indexOf(id);
		if (index >= 0) this.liveOrder.splice(index, 1);
	}

	private emitProgress(progress: TranscriptProgress): void {
		this.revision += 1;
		this.emit({ type: "progress", progress });
	}

	private emitChange(progress?: TranscriptProgress): void {
		this.revision += 1;
		if (progress) this.emit({ type: "progress", progress });
		this.emit({ type: "snapshot" });
	}

	private createToolItem(
		toolCallId: string,
		toolName: string,
		input: JsonValue,
		result: ProjectedToolResult,
		status: "running",
		timestamp: number,
	): RunningToolTranscriptItem;
	private createToolItem(
		toolCallId: string,
		toolName: string,
		input: JsonValue,
		result: ProjectedToolResult,
		status: "complete" | "error",
		timestamp: number,
	): FinishedToolTranscriptItem;
	private createToolItem(
		toolCallId: string,
		toolName: string,
		input: JsonValue,
		result: ProjectedToolResult,
		status: ToolTranscriptItem["status"],
		timestamp: number,
	): ToolTranscriptItem {
		const common = {
			id: `tool-${toolCallId}`,
			role: "tool",
			toolCallId,
			toolName: toolName || "unknown",
			input,
			content: result.content,
			...(result.details === undefined ? {} : { details: result.details }),
			...(result.usage === undefined ? {} : { usage: result.usage }),
			timestamp,
		} as const;
		if (status === "running") return { ...common, status, isError: false };
		if (status === "error") return { ...common, status, isError: true };
		return { ...common, status, isError: false };
	}
}
