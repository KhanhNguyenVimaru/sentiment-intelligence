<script setup lang="ts">
import { ref } from 'vue'

const sentence = ref('')
const result = ref<string | null>(null)
const doneReason = ref<string | null>(null)
const error = ref<string | null>(null)
const loading = ref(false)

const API_URL = 'http://localhost:8000/classify'

const classify = async () => {
  if (!sentence.value.trim() || loading.value) return

  loading.value = true
  error.value = null
  result.value = null
  doneReason.value = null

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sentence: sentence.value }),
    })

    if (!response.ok) throw new Error(`Server returned ${response.status}`)

    const data = await response.json()
    result.value = data.predicted_emotion || 'Unknown'
    doneReason.value = data.done_reason ?? null
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unexpected error'
    error.value = message
  } finally {
    loading.value = false
  }
}
</script>
<template>
  <div class="min-h-screen bg-slate-100 text-slate-800 flex items-center justify-center">
    <div class="mx-auto max-w-5xl px-4 py-10 w-full">
      <div
        class="overflow-hidden rounded-lg border border-linear-to-r from-white via-slate-50 to-slate-100 shadow-xl shadow-slate-400/20 mx-auto"
      >
        <!-- Header -->
        <div class="flex flex-wrap items-center gap-4 border-b border-slate-300 px-6 py-5">
          <div class="space-y-1">
            <p class="text-[11px] uppercase tracking-[0.2em] text-cyan-600">Emotion Classifier</p>
            <h1 class="text-2xl font-bold text-slate-900">Enter a sentence and see the result below</h1>
            <p class="text-sm text-slate-600">
              Frontend calls FastAPI in
              <span class="rounded bg-slate-200 px-2 py-0.5 font-mono text-xs">backend/api_server.py</span>.
            </p>
          </div>
        </div>

        <!-- Form -->
        <div class="space-y-8 px-6 py-6 sm:px-8">
          <form class="space-y-3" @submit.prevent="classify">
            <label class="block text-sm font-semibold text-slate-700 my-4" for="sentence">Sentence to analyze</label>
            <textarea
              id="sentence"
              v-model="sentence"
              rows="4"
              placeholder="Example: I feel so excited about the trip tomorrow!"
              class="w-full rounded-lg my-4 border border-slate-300 bg-white px-4 py-3 text-sm placeholder:text-slate-400 focus:border-cyan-500 focus:outline-none focus:ring-2 focus:ring-cyan-400/30 transition"
            />

            <div class="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <button
                type="submit"
                :disabled="loading || !sentence.trim()"
                class="inline-flex items-center justify-center gap-2 rounded-lg bg-blue-700 px-4 py-3 text-sm font-semibold text-white shadow-lg transition hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <span
                  v-if="loading"
                  class="h-4 w-4 animate-spin rounded-full border-2 border-white/60 border-t-transparent"
                  aria-hidden="true"
                ></span>
                {{ loading ? 'Sending...' : 'Analyze' }}
              </button>

              <span class="text-xs text-slate-500">
                Sent to
                <span class="rounded bg-slate-200 px-2 py-0.5 font-mono text-[11px]">POST /classify</span>
              </span>
            </div>
          </form>

          <!-- Result -->
          <div class="space-y-2 border-t border-slate-300 pt-6">
            <p class="text-[11px] uppercase tracking-[0.2em] text-cyan-600">Result</p>

            <div v-if="loading" class="flex items-center gap-3 text-slate-600">
              <div class="h-2 w-16 rounded-full border-linear-to-r from-cyan-400 via-blue-500 to-cyan-400 animate-pulse"></div>
              <span class="text-sm">Waiting for model response...</span>
            </div>

            <div
              v-else-if="error"
              class="rounded-xl border border-red-400 bg-red-50 px-4 py-3 text-sm text-red-700"
            >
              {{ error }}
            </div>

            <div
              v-else-if="result"
              class="rounded-xl border border-emerald-400 bg-emerald-50 px-4 py-3"
            >
              <div class="text-lg font-extrabold capitalize text-emerald-700">{{ result }}</div>
              <p v-if="doneReason" class="mt-1 text-xs text-emerald-600/80">done_reason: {{ doneReason }}</p>
            </div>

            <div v-else class="text-sm text-slate-400">No result yet.</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
