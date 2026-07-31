import { type Api, clampThinkingLevel, type Model } from "@earendil-works/pi-ai";
import type { ModelMetadata, ModelRef, ThinkingLevel } from "@earendil-works/pi-protocol";
import { PiServerError, toProtocolModelMetadata } from "@earendil-works/pi-server";
import { DEFAULT_THINKING_LEVEL } from "../core/defaults.ts";
import type { ModelRuntime } from "../core/model-runtime.ts";
import type { SettingsManager } from "../core/settings-manager.ts";

export class ServerModelResolver {
	private readonly modelRuntime: ModelRuntime;
	private readonly settingsManager: SettingsManager;
	private defaultModel: ModelRef | undefined;
	private defaultThinkingLevel: ThinkingLevel | undefined;

	constructor(modelRuntime: ModelRuntime, settingsManager: SettingsManager) {
		this.modelRuntime = modelRuntime;
		this.settingsManager = settingsManager;
	}

	setDefaults(options: { model?: ModelRef; thinkingLevel?: ThinkingLevel }): void {
		this.defaultModel = options.model;
		this.defaultThinkingLevel = options.thinkingLevel;
	}

	async list(): Promise<ModelMetadata[]> {
		await this.modelRuntime.getAvailable();
		return this.modelRuntime
			.getModels()
			.map((model) => toProtocolModelMetadata(model, this.modelRuntime.hasConfiguredAuth(model.provider)));
	}

	async resolve(reference?: ModelRef): Promise<Model<Api>> {
		await this.modelRuntime.getAvailable();
		const requested = reference ?? this.defaultModel;
		if (requested) {
			const model = this.modelRuntime.getModel(requested.provider, requested.id);
			if (!model)
				throw new PiServerError("invalid_request", `Could not resolve ${requested.provider}/${requested.id}`);
			return this.requireAuthenticated(model);
		}
		const provider = this.settingsManager.getDefaultProvider();
		const id = this.settingsManager.getDefaultModel();
		const configured = provider && id ? this.modelRuntime.getModel(provider, id) : undefined;
		let model = configured && this.modelRuntime.hasConfiguredAuth(configured.provider) ? configured : undefined;
		if (!model) model = this.modelRuntime.getAvailableSnapshot()[0];
		if (!model) throw new PiServerError("invalid_request", "Could not resolve a default model");
		return this.requireAuthenticated(model);
	}

	resolveAvailable(reference: ModelRef): Model<Api> {
		const model = this.modelRuntime.getModel(reference.provider, reference.id);
		if (!model) throw new PiServerError("invalid_request", `Unknown model: ${reference.provider}/${reference.id}`);
		return this.requireAuthenticated(model);
	}

	private requireAuthenticated(model: Model<Api>): Model<Api> {
		if (!this.modelRuntime.hasConfiguredAuth(model.provider)) {
			throw new PiServerError("invalid_request", `Model is not authenticated: ${model.provider}/${model.id}`);
		}
		return model;
	}

	recoverThinkingLevel(model: Model<Api>, persisted?: ThinkingLevel): ThinkingLevel {
		return persisted === undefined ? this.resolveThinkingLevel(model) : clampThinkingLevel(model, persisted);
	}

	resolveThinkingLevel(model: Model<Api>, requested?: ThinkingLevel): ThinkingLevel {
		const fallback =
			this.defaultThinkingLevel ?? this.settingsManager.getDefaultThinkingLevel() ?? DEFAULT_THINKING_LEVEL;
		const level = requested ?? fallback;
		const clamped = clampThinkingLevel(model, level);
		if (requested !== undefined && clamped !== requested) {
			throw new PiServerError(
				"invalid_request",
				`Thinking level ${requested} is not supported by ${model.provider}/${model.id}`,
			);
		}
		return clamped;
	}
}
