import { GoogleGenerativeAI, type GenerativeModel } from '@google/generative-ai'

const MODEL_NAME = 'gemini-2.5-flash'
const LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'] as const
type EmotionLabel = (typeof LABELS)[number]

const LABEL_SYNONYMS: Record<string, EmotionLabel> = {
  happy: 'joy',
  happiness: 'joy',
  joyful: 'joy',
  ecstatic: 'joy',
  sad: 'sadness',
  depressed: 'sadness',
  anger: 'anger',
  angry: 'anger',
  mad: 'anger',
  furious: 'anger',
  afraid: 'fear',
  fearful: 'fear',
  scared: 'fear',
  terrified: 'fear',
  surprise: 'surprise',
  surprised: 'surprise',
  shocked: 'surprise',
  astonished: 'surprise',
  love: 'love',
  loved: 'love',
  loving: 'love',
}

const MODEL_CONFIG = {
  temperature: 0,
  topP: 1,
  topK: 1,
  maxOutputTokens: 128,
  responseMimeType: 'application/json',
} as const

export interface ClassifyEmotionResult {
  sentence: string
  predictedEmotion: EmotionLabel | null
  doneReason: string | null
  rawResponse: string
}

let cachedKey: string | null = null
let cachedModel: GenerativeModel | null = null

export function getGeminiApiKey(): string {
  const candidate =
    import.meta.env.VITE_GEMINI_API_KEY ??
    import.meta.env.GEMINI_API_KEY ??
    ''

  const trimmed = candidate.trim()
  if (!trimmed) {
    throw new Error('Missing VITE_GEMINI_API_KEY value. Define it in app/.env and restart Vite.')
  }

  return trimmed
}

function ensureModel(): GenerativeModel {
  const apiKey = getGeminiApiKey()
  if (cachedModel && cachedKey === apiKey) return cachedModel

  const client = new GoogleGenerativeAI(apiKey)
  cachedModel = client.getGenerativeModel({
    model: MODEL_NAME,
    generationConfig: MODEL_CONFIG,
  })
  cachedKey = apiKey
  return cachedModel
}

function buildPrompt(sentence: string): string {
  return [
    'Classify the emotion conveyed by the following sentence.',
    `Allowed labels: ${LABELS.join(', ')}.`,
    'Reply strictly in JSON format: {"label": "<label>"}',
    `Sentence: ${sentence}`,
  ].join('\n')
}

function buildBatchPrompt(sentences: string[]): string {
  return [
    'Classify the emotion conveyed by each sentence below.',
    `Allowed labels: ${LABELS.join(', ')}.`,
    'Reply strictly in JSON array format. Each element must look like {"sentence": "<original>", "label": "<label>"}.',
    'Preserve the original order and sentences exactly as provided.',
    'Sentences:',
    ...sentences.map((sentence, index) => `${index + 1}. ${sentence}`),
  ].join('\n')
}

function sanitizeModelText(raw: string): string {
  return raw.replace(/```(?:json)?/gi, '').trim()
}

function tryParseJson(text: string): unknown {
  try {
    return JSON.parse(text)
  } catch {
    return null
  }
}

function extractLabelCandidate(text: string): string {
  const sanitized = sanitizeModelText(text)
  if (!sanitized) return ''

  const direct = tryParseJson(sanitized)
  if (typeof direct === 'string') return direct
  if (direct && typeof direct === 'object') {
    const label = (direct as Record<string, unknown>).label
    if (typeof label === 'string') return label
  }

  const jsonMatch = sanitized.match(/\{[\s\S]*?\}/)
  if (jsonMatch) {
    const parsed = tryParseJson(jsonMatch[0])
    if (parsed && typeof parsed === 'object') {
      const label = (parsed as Record<string, unknown>).label
      if (typeof label === 'string') return label
    }
  }

  return sanitized.split(/\s+/)[0] ?? ''
}

function extractBatchCandidates(text: string): unknown {
  const sanitized = sanitizeModelText(text)
  if (!sanitized) return null

  const parsed = tryParseJson(sanitized)
  if (Array.isArray(parsed)) return parsed

  const start = sanitized.indexOf('[')
  const end = sanitized.lastIndexOf(']')
  if (start !== -1 && end !== -1 && end > start) {
    const sliced = sanitized.slice(start, end + 1)
    const retry = tryParseJson(sliced)
    if (Array.isArray(retry)) return retry
  }

  return null
}

function normalizeLabel(raw: string): EmotionLabel | null {
  const cleaned = raw.toLowerCase().trim()
  if (!cleaned) return null

  const [firstToken = ''] = cleaned.split(/\s+/)
  const candidate = firstToken.trim()
  if (!candidate) return null

  if ((LABELS as readonly string[]).includes(candidate)) {
    return candidate as EmotionLabel
  }
  return LABEL_SYNONYMS[candidate] ?? null
}

export async function classifyEmotion(sentence: string): Promise<ClassifyEmotionResult> {
  const trimmed = sentence?.trim()
  if (!trimmed) {
    throw new Error('Sentence must not be empty.')
  }

  const model = ensureModel()
  const result = await model.generateContent(buildPrompt(trimmed))
  const response = await result.response

  const finishReason = response.candidates?.[0]?.finishReason ?? null
  const rawText = response.text() ?? ''
  const normalized = normalizeLabel(extractLabelCandidate(rawText))

  return {
    sentence: trimmed,
    predictedEmotion: normalized,
    doneReason: finishReason,
    rawResponse: rawText,
  }
}

export async function classifyEmotionBatch(sentences: string[]): Promise<ClassifyEmotionResult[]> {
  const cleaned = sentences.map((s) => s.trim()).filter((s): s is string => Boolean(s))
  if (!cleaned.length) {
    throw new Error('At least one sentence is required.')
  }

  const model = ensureModel()
  const result = await model.generateContent(buildBatchPrompt(cleaned))
  const response = await result.response

  const finishReason = response.candidates?.[0]?.finishReason ?? null
  const rawText = response.text() ?? ''
  const parsed = extractBatchCandidates(rawText)

  if (!Array.isArray(parsed)) {
    throw new Error('Unable to parse batch response from Gemini.')
  }

  return cleaned.map((sentence, index) => {
    const entry = parsed[index]
    let label = ''
    if (entry && typeof entry === 'object') {
      const candidate = (entry as Record<string, unknown>).label
      if (typeof candidate === 'string') {
        label = candidate
      }
    }

    return {
      sentence,
      predictedEmotion: normalizeLabel(label),
      doneReason: finishReason,
      rawResponse: rawText,
    }
  })
}
