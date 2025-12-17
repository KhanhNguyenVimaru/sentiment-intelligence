<script setup lang="ts">
import { ref } from 'vue'
import { classifyEmotion } from '../gemini'

const sentence = ref('')
const prediction = ref<string | null>(null)
const error = ref<string | null>(null)
const loading = ref(false)

const classify = async () => {
  if (!sentence.value.trim() || loading.value) return

  loading.value = true
  error.value = null
  prediction.value = null

  try {
    const response = await classifyEmotion(sentence.value)
    prediction.value = response.predictedEmotion ?? 'Unknown'
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Unexpected error'
  } finally {
    loading.value = false
  }
}
</script>
<template>
  <div class="min-h-screen bg-[#f5f7fb] text-slate-800 flex items-center justify-center px-4 py-8">
    <div class="w-full max-w-xl rounded-lg border border-slate-200 bg-white shadow-[0_20px_60px_rgba(15,23,42,0.08)] overflow-hidden">
      <div class="px-8 py-6 border-b border-slate-100 bg-slate-100">
        <p class="text-xs font-semibold uppercase tracking-[0.3em] text-slate-500">Emotion Classifier</p>
        <h1 class="mt-2 text-2xl font-bold text-slate-900">Gemini-powered analysis</h1>
        <p class="text-sm text-slate-600">Enter any sentence to predict the dominant emotion.</p>
      </div>

      <form class="px-8 py-6 space-y-4" @submit.prevent="classify">
        <label class="text-sm font-medium text-slate-600" for="sentence">Sentence</label>
        <textarea
          id="sentence"
          v-model="sentence"
          rows="4"
          placeholder="Example: I feel so excited about the trip tomorrow!"
          class="w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm placeholder:text-slate-400 focus:border-sky-400 focus:outline-none focus:ring-4 focus:ring-sky-200 transition"
        />

        <button
          type="submit"
          :disabled="loading || !sentence.trim()"
          class="w-full inline-flex items-center justify-center gap-2 rounded-2xl bg-sky-600 px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-sky-600/30 transition hover:bg-sky-700 disabled:cursor-not-allowed disabled:opacity-60"
        >
          <span
            v-if="loading"
            class="h-4 w-4 animate-spin rounded-full border-2 border-white/60 border-t-transparent"
            aria-hidden="true"
          ></span>
          {{ loading ? 'Analyzing...' : 'Analyze sentence' }}
        </button>

        <p class="text-[11px] text-center text-slate-400">Uses Gemini API key from <code>.env</code></p>
      </form>

      <div class="px-8 py-6 border-t border-slate-100 bg-slate-50">
        <p class="text-xs font-semibold uppercase tracking-[0.4em] text-slate-500 mb-3">Result</p>

        <div v-if="loading" class="flex items-center gap-3 text-slate-500 text-sm">
          <div class="h-2 w-16 rounded-full bg-linear-to-r from-sky-400 via-indigo-500 to-sky-400 animate-pulse"></div>
          Fetching prediction...
        </div>

        <div v-else-if="error" class="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {{ error }}
        </div>

        <div
          v-else-if="prediction"
          class="rounded-2xl border border-emerald-200 bg-white px-4 py-5 text-center shadow-inner"
        >
          <p class="text-xs uppercase tracking-[0.3em] text-emerald-500">Predicted emotion</p>
          <p class="mt-2 text-3xl font-black text-emerald-700 capitalize">{{ prediction }}</p>
        </div>

        <p v-else class="text-sm text-slate-400">No prediction yet.</p>
      </div>
    </div>
  </div>
</template>
