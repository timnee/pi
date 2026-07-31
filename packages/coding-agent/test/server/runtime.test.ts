import { join } from "node:path";
import { NodeExecutionEnv } from "@earendil-works/pi-agent-core/node";
import { fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import type { TranscriptProgress } from "@earendil-works/pi-protocol";
import lockfile, { type LockOptions } from "proper-lockfile";
import { describe, expect, test, vi } from "vitest";
import { CodingAgentServerBackend } from "../../src/server/backend.ts";
import { abortableResponse, createServerBackendFixture, removeServerBackendFixture } from "./fixture.ts";

describe("coding-agent session runtime", () => {
	test("propagates lock status check failures", async () => {
		const fixture = await createServerBackendFixture();
		const runtime = await fixture.backend.createSession({ id: "server-lock-check", cwd: fixture.cwd });
		await runtime.dispose();
		const checkSpy = vi.spyOn(lockfile, "check").mockRejectedValue(new Error("lock check failed"));
		try {
			await expect(fixture.backend.listSessions()).rejects.toThrow(/Failed to read coding-agent session/);
		} finally {
			checkSpy.mockRestore();
			await removeServerBackendFixture(fixture);
		}
	});
	test("terminates an active runtime without persisting after its session lock is compromised", async () => {
		const fixture = await createServerBackendFixture();
		const realLock = lockfile.lock.bind(lockfile);
		let compromise: LockOptions["onCompromised"];
		let forceRelease: (() => Promise<void>) | undefined;
		const lockSpy = vi.spyOn(lockfile, "lock").mockImplementation(async (file, options) => {
			compromise = options?.onCompromised;
			forceRelease = await realLock(file, options);
			return forceRelease;
		});
		let runtime: Awaited<ReturnType<typeof fixture.backend.createSession>> | undefined;
		try {
			runtime = await fixture.backend.createSession({
				id: "server-compromised-lock",
				cwd: fixture.cwd,
				model: { provider: fixture.faux.provider.id, id: "faux-reasoning" },
			});
			const pendingResponse = abortableResponse();
			fixture.faux.setResponses([pendingResponse.response]);
			const prompt = runtime.prompt({ text: "start" });
			expect(runtime.getPhase()).toBe("turn");
			await pendingResponse.started;
			await forceRelease?.();
			compromise?.(new Error("lock ownership lost"));

			await expect(prompt).rejects.toMatchObject({ code: "session_locked" });
			await expect(runtime.snapshot()).rejects.toMatchObject({ code: "session_locked" });
			await runtime.dispose();
			const secondBackend = await CodingAgentServerBackend.create({
				defaultCwd: fixture.cwd,
				sessionRoot: join(fixture.root, "sessions"),
				modelRuntime: fixture.modelRuntime,
				settingsManager: fixture.settingsManager,
			});
			const reopened = await secondBackend.openSession("server-compromised-lock");
			try {
				expect((await reopened.snapshot()).transcript.some((item) => item.role === "assistant")).toBe(false);
			} finally {
				await reopened.dispose();
			}
		} finally {
			lockSpy.mockRestore();
			await runtime?.dispose();
			await removeServerBackendFixture(fixture);
		}
	});
	test("propagates lock release failure and allows disposal retry", async () => {
		const fixture = await createServerBackendFixture();
		const realLock = lockfile.lock.bind(lockfile);
		let releaseAttempts = 0;
		const lockSpy = vi.spyOn(lockfile, "lock").mockImplementation(async (file, options) => {
			const release = await realLock(file, options);
			return async () => {
				releaseAttempts += 1;
				if (releaseAttempts === 1) throw new Error("release failed");
				await release();
			};
		});
		let runtime: Awaited<ReturnType<typeof fixture.backend.createSession>> | undefined;
		try {
			runtime = await fixture.backend.createSession({ id: "server-release-retry", cwd: fixture.cwd });
			await expect(runtime.dispose()).rejects.toThrow("release failed");
			await expect(runtime.dispose()).resolves.toBeUndefined();
			expect(releaseAttempts).toBe(2);
		} finally {
			lockSpy.mockRestore();
			await runtime?.dispose();
			await removeServerBackendFixture(fixture);
		}
	});
	test("exposes accepted steering text to every session snapshot", async () => {
		const fixture = await createServerBackendFixture();
		const runtime = await fixture.backend.createSession({
			id: "server-visible-steer",
			cwd: fixture.cwd,
			model: { provider: fixture.faux.provider.id, id: "faux-reasoning" },
		});
		try {
			const pendingResponse = abortableResponse();
			fixture.faux.setResponses([pendingResponse.response]);
			const prompt = runtime.prompt({ text: "start" });
			expect(runtime.getPhase()).toBe("turn");
			await pendingResponse.started;
			await runtime.steer({ text: "adjust the approach" });
			expect(await runtime.snapshot()).toMatchObject({
				queuedSteerCount: 1,
				queuedSteer: [{ role: "user", content: [{ type: "text", text: "adjust the approach" }] }],
			});
			await runtime.abort();
			await prompt;
		} finally {
			await runtime.dispose();
			await removeServerBackendFixture(fixture);
		}
	});
	test("rejects concurrent structural mutations instead of queueing them", async () => {
		const fixture = await createServerBackendFixture();
		const runtime = await fixture.backend.createSession({
			id: "server-concurrent-mutation",
			cwd: fixture.cwd,
			model: { provider: fixture.faux.provider.id, id: "faux-reasoning" },
		});
		try {
			const changingModel = runtime.setModel({ provider: fixture.faux.provider.id, id: "faux-plain" });
			await expect(runtime.setThinking("medium")).rejects.toMatchObject({ code: "busy" });
			await expect(runtime.prompt({ text: "must not queue" })).rejects.toMatchObject({ code: "busy" });
			await expect(runtime.abort()).rejects.toMatchObject({ code: "busy" });
			await changingModel;
		} finally {
			await runtime.dispose();
			await removeServerBackendFixture(fixture);
		}
	});
	test("terminates after a model change is only partially persisted", async () => {
		const fixture = await createServerBackendFixture();
		const runtime = await fixture.backend.createSession({
			id: "server-partial-model-change",
			cwd: fixture.cwd,
			model: { provider: fixture.faux.provider.id, id: "faux-reasoning" },
			thinkingLevel: "high",
		});
		const originalAppendFile = NodeExecutionEnv.prototype.appendFile;
		const appendSpy = vi.spyOn(NodeExecutionEnv.prototype, "appendFile");
		appendSpy.mockImplementationOnce(function (this: NodeExecutionEnv, ...args) {
			return originalAppendFile.apply(this, args);
		});
		appendSpy.mockRejectedValueOnce(new Error("thinking write failed"));
		try {
			await expect(runtime.setModel({ provider: fixture.faux.provider.id, id: "faux-plain" })).rejects.toMatchObject(
				{
					code: "invalid_request",
					message: "Session failed to set model",
					cause: expect.objectContaining({ message: expect.stringContaining("thinking write failed") }),
				},
			);
			await expect(runtime.snapshot()).rejects.toMatchObject({
				code: "invalid_request",
				message: "Session failed to set model",
			});
		} finally {
			appendSpy.mockRestore();
			await runtime.dispose();
			await removeServerBackendFixture(fixture);
		}
	});
	test("streams live bash output into progress and the final snapshot", async () => {
		const fixture = await createServerBackendFixture();
		const runtime = await fixture.backend.createSession({
			id: "server-session-bash",
			cwd: fixture.cwd,
			model: { provider: fixture.faux.provider.id, id: "faux-reasoning" },
			thinkingLevel: "low",
		});
		try {
			fixture.faux.setResponses([
				fauxAssistantMessage(
					fauxToolCall("bash", { command: "printf first; sleep 0.2; printf second" }, { id: "bash-call" }),
					{ stopReason: "toolUse" },
				),
				fauxAssistantMessage("done"),
			]);
			const progress: TranscriptProgress[] = [];
			runtime.subscribe((event) => {
				if (event.type === "progress") progress.push(event.progress);
			});
			await runtime.prompt({ text: "run it" });

			const toolUpdates = progress.filter(
				(event): event is Extract<TranscriptProgress, { type: "item_updated" }> =>
					event.type === "item_updated" && event.item.role === "tool",
			);
			expect(
				toolUpdates.some((event) =>
					event.item.content.some((part) => part.type === "text" && part.text.includes("first")),
				),
			).toBe(true);
			const final = await runtime.snapshot();
			expect(final.transcript.map((item) => item.role)).toEqual(["user", "assistant", "tool", "assistant"]);
			const tool = final.transcript.find((item) => item.role === "tool");
			expect(tool).toMatchObject({
				toolCallId: "bash-call",
				toolName: "bash",
				input: { command: "printf first; sleep 0.2; printf second" },
				status: "complete",
				isError: false,
			});
			if (!tool || tool.role !== "tool") throw new Error("Expected tool transcript item");
			const output = tool.content.flatMap((part) => (part.type === "text" ? [part.text] : [])).join("");
			expect(output).toContain("first");
			expect(output).toContain("second");
		} finally {
			await runtime.dispose();
			await removeServerBackendFixture(fixture);
		}
	});
	test("terminates when executed tool input cannot be represented by the protocol", async () => {
		const fixture = await createServerBackendFixture();
		const runtime = await fixture.backend.createSession({
			id: "server-invalid-tool-input",
			cwd: fixture.cwd,
			model: { provider: fixture.faux.provider.id, id: "faux-reasoning" },
		});
		try {
			let projectionError: Error | undefined;
			runtime.subscribe((event) => {
				if (event.type === "error") projectionError = event.error;
			});
			fixture.faux.setResponses([
				fauxAssistantMessage(fauxToolCall("read", { path: Number.NaN }), { stopReason: "toolUse" }),
			]);
			await expect(runtime.prompt({ text: "trigger invalid input" })).rejects.toMatchObject({
				code: "invalid_request",
				message: "Session protocol projection failed",
			});
			expect(projectionError?.cause).toBeInstanceOf(TypeError);
			await expect(runtime.snapshot()).rejects.toMatchObject({ code: "invalid_request" });
		} finally {
			await runtime.dispose();
			await removeServerBackendFixture(fixture);
		}
	});
});
