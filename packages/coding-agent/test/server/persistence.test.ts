import { randomUUID } from "node:crypto";
import { appendFile, readdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { NodeExecutionEnv } from "@earendil-works/pi-agent-core/node";
import { describe, expect, test, vi } from "vitest";
import { createServerBackendFixture, removeServerBackendFixture } from "./fixture.ts";

describe("coding-agent server persistence", () => {
	test("fails session listing when persisted JSONL is unreadable", async () => {
		const fixture = await createServerBackendFixture();
		const runtime = await fixture.backend.createSession({
			id: "server-corrupt-session",
			cwd: fixture.cwd,
		});
		await runtime.dispose();
		const files = await readdir(join(fixture.root, "sessions"), { recursive: true });
		const sessionFile = files.find((file) => file.endsWith(".jsonl"));
		if (!sessionFile) throw new Error("Expected persisted JSONL session");
		await appendFile(join(fixture.root, "sessions", sessionFile), "{invalid json\n");

		try {
			await expect(fixture.backend.listSessions()).rejects.toThrow(/Failed to read coding-agent session/);
		} finally {
			await removeServerBackendFixture(fixture);
		}
	});
	test("surfaces corrupt session headers instead of omitting them", async () => {
		const fixture = await createServerBackendFixture();
		const runtime = await fixture.backend.createSession({ id: "server-corrupt-header", cwd: fixture.cwd });
		await runtime.dispose();
		const files = await readdir(join(fixture.root, "sessions"), { recursive: true });
		const sessionFile = files.find((file) => file.endsWith(".jsonl"));
		if (!sessionFile) throw new Error("Expected persisted JSONL session");
		await writeFile(join(fixture.root, "sessions", sessionFile), "{invalid header\n");
		try {
			await expect(fixture.backend.listSessions()).rejects.toThrow(/Invalid JSONL session file/);
			await expect(fixture.backend.openSession("server-corrupt-header")).rejects.toThrow(
				/Invalid JSONL session file/,
			);
		} finally {
			await removeServerBackendFixture(fixture);
		}
	});
	test("rejects invalid persisted creation timestamps", async () => {
		const fixture = await createServerBackendFixture();
		const runtime = await fixture.backend.createSession({ id: "server-invalid-created-at", cwd: fixture.cwd });
		await runtime.dispose();
		const files = await readdir(join(fixture.root, "sessions"), { recursive: true });
		const relative = files.find((file) => file.endsWith(".jsonl"));
		if (!relative) throw new Error("Expected persisted JSONL session");
		const sessionFile = join(fixture.root, "sessions", relative);
		const lines = (await readFile(sessionFile, "utf8")).split("\n");
		const header = JSON.parse(lines[0]!) as Record<string, unknown>;
		header.timestamp = "not-a-date";
		lines[0] = JSON.stringify(header);
		await writeFile(sessionFile, lines.join("\n"));
		try {
			await expect(fixture.backend.listSessions()).rejects.toMatchObject({
				cause: expect.objectContaining({ message: expect.stringMatching(/Invalid session creation timestamp/) }),
			});
			await expect(fixture.backend.openSession("server-invalid-created-at")).rejects.toThrow(
				/Invalid session creation timestamp/,
			);
		} finally {
			await removeServerBackendFixture(fixture);
		}
	});
	test("rejects invalid persisted entry timestamps", async () => {
		const fixture = await createServerBackendFixture();
		const runtime = await fixture.backend.createSession({ id: "server-invalid-entry-time", cwd: fixture.cwd });
		await runtime.dispose();
		const files = await readdir(join(fixture.root, "sessions"), { recursive: true });
		const relative = files.find((file) => file.endsWith(".jsonl"));
		if (!relative) throw new Error("Expected persisted JSONL session");
		const sessionFile = join(fixture.root, "sessions", relative);
		const lines = (await readFile(sessionFile, "utf8")).split("\n");
		const entry = JSON.parse(lines[1]!) as Record<string, unknown>;
		entry.timestamp = "not-a-date";
		lines[1] = JSON.stringify(entry);
		await writeFile(sessionFile, lines.join("\n"));
		try {
			await expect(fixture.backend.listSessions()).rejects.toMatchObject({
				cause: expect.objectContaining({
					message: expect.stringMatching(/Invalid timestamp on session entry/),
				}),
			});
			await expect(fixture.backend.openSession("server-invalid-entry-time")).rejects.toThrow(
				/Invalid timestamp on session entry/,
			);
		} finally {
			await removeServerBackendFixture(fixture);
		}
	});
	test("removes a partially persisted session when creation fails", async () => {
		const fixture = await createServerBackendFixture();
		const appendSpy = vi
			.spyOn(NodeExecutionEnv.prototype, "appendFile")
			.mockRejectedValueOnce(new Error("write failed"));
		try {
			await expect(fixture.backend.createSession({ id: "server-partial-create", cwd: fixture.cwd })).rejects.toThrow(
				"write failed",
			);
			expect(await fixture.backend.listSessions()).toEqual([]);
			await expect(fixture.backend.openSession("server-partial-create")).rejects.toMatchObject({
				code: "not_found",
			});
		} finally {
			appendSpy.mockRestore();
			await removeServerBackendFixture(fixture);
		}
	});
	test("repairs a persisted thinking level that is unsupported by its model", async () => {
		const fixture = await createServerBackendFixture();
		const runtime = await fixture.backend.createSession({
			id: "server-interrupted-model-change",
			cwd: fixture.cwd,
			model: { provider: fixture.faux.provider.id, id: "faux-reasoning" },
			thinkingLevel: "high",
		});
		await runtime.dispose();
		const files = await readdir(join(fixture.root, "sessions"), { recursive: true });
		const relative = files.find((file) => file.endsWith(".jsonl"));
		if (!relative) throw new Error("Expected persisted JSONL session");
		const sessionFile = join(fixture.root, "sessions", relative);
		const lines = (await readFile(sessionFile, "utf8")).trimEnd().split("\n");
		const previous = JSON.parse(lines.at(-1)!) as { id: string };
		await appendFile(
			sessionFile,
			`${JSON.stringify({
				type: "model_change",
				id: randomUUID(),
				parentId: previous.id,
				timestamp: new Date().toISOString(),
				provider: fixture.faux.provider.id,
				modelId: "faux-plain",
			})}\n`,
		);
		try {
			const reopened = await fixture.backend.openSession("server-interrupted-model-change");
			try {
				expect(await reopened.snapshot()).toMatchObject({
					model: { provider: fixture.faux.provider.id, id: "faux-plain" },
					thinkingLevel: "off",
				});
			} finally {
				await reopened.dispose();
			}
			const repaired = await fixture.backend.openSession("server-interrupted-model-change");
			try {
				expect((await repaired.snapshot()).thinkingLevel).toBe("off");
			} finally {
				await repaired.dispose();
			}
		} finally {
			await removeServerBackendFixture(fixture);
		}
	});
});
