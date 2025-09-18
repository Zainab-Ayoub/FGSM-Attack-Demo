'use client';

import { useMemo, useState } from 'react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Page() {
  const [file, setFile] = useState<File | null>(null);
  const [epsilon, setEpsilon] = useState<number>(0.1);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const canSubmit = useMemo(() => !!file && !loading, [file, loading]);

  const onSubmit = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append('image', file);
      form.append('epsilon', String(epsilon));
      const res = await fetch(`${API_URL}/attack`, {
        method: 'POST',
        body: form,
      });
      if (!res.ok) throw new Error(`Request failed: ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (e: any) {
      setError(e?.message || 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main>
      <h1 style={{ marginBottom: 16 }}>FGSM Attack Demo</h1>
      <div style={{ display: 'grid', gap: 12, maxWidth: 560 }}>
        <input
          type="file"
          accept="image/png,image/jpeg"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />
        <label>
          Epsilon: {epsilon.toFixed(2)}
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={epsilon}
            onChange={(e) => setEpsilon(Number(e.target.value))}
            style={{ width: '100%' }}
          />
        </label>
        <button onClick={onSubmit} disabled={!canSubmit}>
          {loading ? 'Running…' : 'Run Attack'}
        </button>
      </div>

      {error && (
        <p style={{ color: 'crimson', marginTop: 12 }}>Error: {error}</p>
      )}

      {result && (
        <section style={{ marginTop: 24 }}>
          <h2>Results</h2>
          <p>
            <strong>Attack success:</strong> {String(result.attack_success)}
          </p>
          <p>
            <strong>Clean prediction:</strong> {result.clean_prediction}
            {result.clean_label ? ` (${result.clean_label})` : ''}
          </p>
          <p>
            <strong>Adversarial prediction:</strong> {result.adversarial_prediction}
            {result.adversarial_label ? ` (${result.adversarial_label})` : ''}
          </p>

          <div style={{ display: 'flex', gap: 16, marginTop: 12 }}>
            <figure>
              <figcaption>Clean</figcaption>
              <img
                src={`data:image/png;base64,${result.clean_image_base64}`}
                alt="Clean"
                style={{ maxWidth: 256, border: '1px solid #ddd' }}
              />
            </figure>
            <figure>
              <figcaption>Adversarial</figcaption>
              <img
                src={`data:image/png;base64,${result.adversarial_image_base64}`}
                alt="Adversarial"
                style={{ maxWidth: 256, border: '1px solid #ddd' }}
              />
            </figure>
          </div>
        </section>
      )}
    </main>
  );
}


