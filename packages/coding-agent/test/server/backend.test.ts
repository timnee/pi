import { join } from "node:path";
import { fauxAssistantMessage } from "@earendil-works/pi-ai";
import type { TranscriptProgress } from "@earendil-works/pi-protocol";
import { describe, expect, test } from "vitest";
import { CodingAgentServerBackend, toPiServerError } from "../../src/server/backend.ts";
import { createServerBackendFixture, removeServerBackendFixture } from "./fixture.ts";

describe("coding-agent server backend", () => {
	test("normalizes snapshots, restores model/thinking, and holds an exclusive session lock", async () => {
		const fixture = await createServerBackendFixture();
		let runtime = await fixture.backend.createSession({
			id: "server-session-1",
			cwd: fixture.cwd,
			name: "Backend test",
			model: { provider: fixture.faux.provider.id, id: "faux-reasoning" },
			thinkingLevel: "high",
		});
		try {
			const initial = await runtime.snapshot();
			expect(initial).toMatchObject({
				id: "server-session-1",
				name: "Backend test",
				cwd: fixture.cwd,
				phase: "idle",
				model: { provider: fixture.faux.provider.id, id: "faux-reasoning" },
				thinkingLevel: "high",
				revision: 0,
				transcript: [],
				locked: true,
			});

			const models = await fixture.backend.listModels();
			expect(models).toContainEqual(
				expect.objectContaining({
					provider: fixture.faux.provider.id,
					id: "faux-reasoning",
					authenticated: true,
					supportedThinkingLevels: expect.arrayContaining(["off", "high"]),
				}),
			);
			const summaries = await fixture.backend.listSessions();
			expect(summaries).toContainEqual(
				expect.objectContaining({ id: "server-session-1", name: "Backend test", locked: true }),
			);

			const secondBackend = await CodingAgentServerBackend.create({
				defaultCwd: fixture.cwd,
				sessionRoot: join(fixture.root, "sessions"),
				modelRuntime: fixture.modelRuntime,
				settingsManager: fixture.settingsManager,
			});
			await expect(secondBackend.openSession("server-session-1")).rejects.toMatchObject({
				code: "session_locked",
			});

			fixture.faux.setResponses([fauxAssistantMessage("normalized response")]);
			const progress: TranscriptProgress[] = [];
			runtime.subscribe(() => {
				throw new Error("network subscriber failed");
			});
			runtime.subscribe((event) => {
				if (event.type === "progress") progress.push(event.progress);
			});
			await runtime.prompt({ text: "hello" });
			const completed = await runtime.snapshot();
			expect(completed.revision).toBeGreaterThan(initial.revision);
			expect(completed.phase).toBe("idle");
			expect(completed.transcript.map((item) => item.role)).toEqual(["user", "assistant"]);
			expect(completed.transcript[1]).toMatchObject({
				role: "assistant",
				content: [{ type: "text", text: "normalized response" }],
				status: "complete",
				stopReason: "stop",
			});
			expect(completed.transcript.some((item) => "parentId" in item)).toBe(false);
			expect(progress.some((event) => event.type === "assistant_delta")).toBe(true);

			await runtime.setModel({ provider: fixture.faux.provider.id, id: "faux-plain" });
			expect(await runtime.snapshot()).toMatchObject({
				model: { provider: fixture.faux.provider.id, id: "faux-plain" },
				thinkingLevel: "off",
			});
			await runtime.setModel({ provider: fixture.faux.provider.id, id: "faux-reasoning" });
			await runtime.setThinking("high");

			await runtime.dispose();
			const persistedSummary = (await secondBackend.listSessions()).find(
				(summary) => summary.id === "server-session-1",
			);
			expect(persistedSummary).toBeDefined();
			runtime = await secondBackend.openSession("server-session-1");
			const restored = await runtime.snapshot();
			expect(restored.updatedAt).toBe(persistedSummary?.updatedAt);
			expect(restored.model).toEqual(initial.model);
			expect(restored.thinkingLevel).toBe("high");
			expect(restored.transcript).toEqual(completed.transcript);
		} finally {
			await runtime.dispose();
			await removeServerBackendFixture(fixture);
		}
	});
	test("preserves unexpected operational errors for boundary-safe handling", () => {
		const operational = new Error("private filesystem detail");
		expect(toPiServerError(operational)).toBe(operational);
	});
	test("applies server-only defaults to newly created sessions", async () => {
		const fixture = await createServerBackendFixture();
		fixture.backend.setDefaultSessionOptions({
			model: { provider: fixture.faux.provider.id, id: "faux-plain" },
			thinkingLevel: "off",
		});
		const runtime = await fixture.backend.createSession({ id: "server-defaults", cwd: fixture.cwd });
		try {
			expect(await runtime.snapshot()).toMatchObject({
				model: { provider: fixture.faux.provider.id, id: "faux-plain" },
				thinkingLevel: "off",
			});
		} finally {
			await runtime.dispose();
			await removeServerBackendFixture(fixture);
		}
	});
	test("maps unsupported model and thinking requests to protocol-safe errors", async () => {
		const fixture = await createServerBackendFixture();
		try {
			await expect(
				fixture.backend.createSession({
					id: "bad-model",
					cwd: fixture.cwd,
					model: { provider: "missing", id: "missing" },
				}),
			).rejects.toMatchObject({
				name: "PiServerError",
				code: "invalid_request",
				message: expect.stringContaining("Could not resolve missing/missing"),
			});
			await expect(
				fixture.backend.createSession({
					id: "bad-thinking",
					cwd: fixture.cwd,
					model: { provider: fixture.faux.provider.id, id: "faux-reasoning" },
					thinkingLevel: "max",
				}),
			).rejects.toMatchObject({
				name: "PiServerError",
				code: "invalid_request",
				message: expect.stringContaining("not supported"),
			});
		} finally {
			await removeServerBackendFixture(fixture);
		}
	});
});
