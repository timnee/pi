import { AgentHarnessError, SessionError } from "@earendil-works/pi-agent-core";
import { PiServerError } from "@earendil-works/pi-server";

export function toPiServerError(error: unknown): Error {
	if (error instanceof PiServerError) return error;
	if (error instanceof AgentHarnessError) {
		if (error.code === "busy") return new PiServerError("busy", error.message);
		if (error.code === "invalid_argument") return new PiServerError("invalid_request", error.message);
		if (error.code === "session" && error.cause instanceof SessionError && error.cause.code === "not_found") {
			return new PiServerError("not_found", error.message);
		}
	}
	if (error instanceof SessionError && error.code === "not_found") {
		return new PiServerError("not_found", error.message);
	}
	return error instanceof Error ? error : new Error(String(error));
}
