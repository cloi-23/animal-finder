import { useEffect, useState } from "react";
import { AnimalAI } from "../ai/animalAi";
import { IonButton, IonContent, IonPage, IonText } from "@ionic/react";
import { useNavigate } from "react-router";
import { collectionService } from "../database/collectionService";
import { getBreedAvatar } from "./breedAvatars";
import { getBreedExplanation } from "./breedExplanations";
import "./Home.css";

export type PredictionSummary = {
  animal: string;
  breed: string;
  confidencePercent: number;
  isReliable: boolean;
  warning: string;
  topPredictions: Array<{
    label: string;
    confidence: number;
    confidencePercent: number;
  }>;
};

export function getPredictionSummary(result: {
  category?: string;
  name?: string | null;
  confidence?: number | null;
  predictions?: Array<{ label?: string | null; confidence?: number | null }>;
}): PredictionSummary {
  const category =
    result.category && ["Dog", "Cat"].includes(result.category)
      ? result.category
      : "Unknown";
  const confidence =
    typeof result.confidence === "number" ? result.confidence : 0;
  const confidencePercent = Number((confidence * 100).toFixed(2));
  const rawBreed =
    result.name && result.name !== "Unknown" && result.name !== "Unknown animal"
      ? result.name
      : "Unknown";
  const isReliable =
    confidence >= 0.5 && rawBreed !== "Unknown" && category !== "Unknown";
  const breed = isReliable ? rawBreed : "Unknown";

  const topPredictions = (result.predictions ?? [])
    .slice(0, 5)
    .map((prediction) => {
      const label =
        prediction.label &&
        prediction.label !== "Unknown" &&
        prediction.label !== "Unknown animal"
          ? prediction.label
          : "Unknown";
      const score =
        typeof prediction.confidence === "number" ? prediction.confidence : 0;

      return {
        label,
        confidence: score,
        confidencePercent: Number((score * 100).toFixed(2)),
      };
    })
    .filter((prediction) => prediction.label !== "Unknown");

  return {
    animal: category,
    breed,
    confidencePercent,
    isReliable,
    warning: isReliable
      ? ""
      : "⚠️ Breed confidence is too low for a reliable breed prediction.",
    topPredictions,
  };
}

const Home: React.FC = () => {
  const navigate = useNavigate();
  const [category, setCategory] = useState<"Dog" | "Cat">("Dog");
  const [error, setError] = useState("");
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [identifying, setIdentifying] = useState(false);
  const [predictionSummary, setPredictionSummary] =
    useState<PredictionSummary | null>(null);
  const [displayedConfidence, setDisplayedConfidence] = useState(0);

  useEffect(() => {
    AnimalAI.modelInfo().catch((modelError) =>
      console.error("Animal AI model unavailable", modelError),
    );
  }, []);

  useEffect(() => {
    return () => {
      if (selectedImage) {
        URL.revokeObjectURL(selectedImage);
      }
    };
  }, [selectedImage]);

  useEffect(() => {
    if (identifying || !predictionSummary) {
      setDisplayedConfidence(0);
      return;
    }

    const target = predictionSummary.confidencePercent;
    const startedAt = performance.now();
    const duration = 900;
    let animationFrame = 0;

    const animateConfidence = (timestamp: number) => {
      const progress = Math.min(1, (timestamp - startedAt) / duration);
      const easedProgress = 1 - Math.pow(1 - progress, 3);
      setDisplayedConfidence(target * easedProgress);

      if (progress < 1) {
        animationFrame = requestAnimationFrame(animateConfidence);
      }
    };

    animationFrame = requestAnimationFrame(animateConfidence);

    return () => cancelAnimationFrame(animationFrame);
  }, [identifying, predictionSummary]);

  const handlePhotoSelected = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setError("");
    setPredictionSummary(null);
    setSelectedImage(URL.createObjectURL(file));
    setIdentifying(true);

    try {
      const image = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () =>
          typeof reader.result === "string"
            ? resolve(reader.result)
            : reject(new Error("Unable to read image"));
        reader.onerror = () =>
          reject(reader.error ?? new Error("Unable to read image"));
        reader.readAsDataURL(file);
      });
      const result = await AnimalAI.classify({ image, category });
      console.log("AnimalAI classify result:", JSON.stringify(result));

      const summary = getPredictionSummary(result);
      setPredictionSummary(summary);
      // Track discovered breed in collection
      if (summary.isReliable && summary.breed !== "Unknown") {
        collectionService.recordBreedDiscovery(
          summary.breed,
          summary.animal as "Dog" | "Cat",
          image,
        );
      }
    } catch (classificationError) {
      console.error("Animal AI classification failed:", classificationError);
      setError(
        classificationError instanceof Error
          ? classificationError.message
          : "Unable to identify the animal.",
      );
    } finally {
      setIdentifying(false);
      event.target.value = "";
    }
  };

  const topPredictions = predictionSummary?.topPredictions ?? [];
  const topBreed = topPredictions[0];
  const secondBreed = topPredictions[1];
  const topConfidence = topBreed?.confidencePercent ?? 0;
  const secondConfidence = secondBreed?.confidencePercent ?? 0;
  const breedExplanation = getBreedExplanation(
    predictionSummary?.animal ?? "Unknown",
    topPredictions,
  );
  const highConfidenceCategory =
    predictionSummary &&
    predictionSummary.confidencePercent >= 80 &&
    (predictionSummary.animal === "Cat" || predictionSummary.animal === "Dog")
      ? predictionSummary.animal
      : null;
  const categoryMismatch =
    highConfidenceCategory && highConfidenceCategory !== category;

  return (
    <IonPage>
      <IonContent fullscreen>
        <main className="scanner-home">
          <header className="scanner-header">
            <div className="header-content">
              <span>{category.toUpperCase()} SCANNER</span>
              <IonButton
                fill="clear"
                size="small"
                onClick={() => navigate("/collection")}
                className="collection-button"
              >
                📚
              </IonButton>
            </div>
          </header>

          <div className="category-selector">
            <button
              className={`category-btn ${category === "Dog" ? "active" : ""}`}
              disabled={identifying}
              onClick={() => setCategory("Dog")}
            >
              🐕 Dogs
            </button>
            <button
              className={`category-btn ${category === "Cat" ? "active" : ""}`}
              disabled={identifying}
              onClick={() => setCategory("Cat")}
            >
              🐈 Cats
            </button>
          </div>

          <section className="scanner-panel">
            <h2>Your Result</h2>
            {identifying && (
              <p className="identifying-status">
                Identifying animal
                <span className="loading-dots" aria-hidden="true">
                  ...
                </span>
              </p>
            )}

            <div
              className={`result-ring-wrap ${identifying ? "is-scanning" : ""}`}
            >
              <div
                className="result-ring"
                style={{
                  background: `conic-gradient(
                    #1ea6b4 0 ${topConfidence}%,
                    #1f5b66 ${topConfidence}% ${Math.min(
                      100,
                      topConfidence + secondConfidence,
                    )}%,
                    #d8d8d8 ${Math.min(
                      100,
                      topConfidence + secondConfidence,
                    )}% 100%
                  )`,
                }}
              >
                <div className="ring-center">
                  {identifying ? (
                    <span className="ring-scanning" aria-label="Scanning">
                      ...
                    </span>
                  ) : predictionSummary ? (
                    <strong
                      aria-label={`${displayedConfidence.toFixed(1)} percent confidence`}
                    >
                      {displayedConfidence.toFixed(1)}%
                    </strong>
                  ) : (
                    <span className="ring-ready">Ready</span>
                  )}
                </div>
              </div>

              {selectedImage ? (
                <div className="ring-avatar ring-avatar-main">
                  <img src={selectedImage} alt="Pet result" />
                </div>
              ) : (
                <div className="ring-avatar ring-avatar-main ring-avatar-placeholder">
                  {topBreed ? (
                    <img
                      src={getBreedAvatar(topBreed.label)}
                      alt={topBreed.label}
                    />
                  ) : (
                    <span aria-hidden="true">?</span>
                  )}
                </div>
              )}

              {secondBreed && (
                <div className="ring-avatar ring-avatar-secondary">
                  <img
                    src={getBreedAvatar(secondBreed.label)}
                    alt={secondBreed.label}
                  />
                </div>
              )}
            </div>

            <div className="result-list">
              {topPredictions.length > 0 ? (
                topPredictions.slice(0, 3).map((breed, index) => (
                  <div
                    className={`result-card ${index === 0 ? "result-card-primary" : ""}`}
                    key={breed.label}
                  >
                    <div className="result-card-avatar">
                      {index === 0 && selectedImage ? (
                        <img src={selectedImage} alt="Top breed" />
                      ) : (
                        <img
                          src={getBreedAvatar(breed.label)}
                          alt={breed.label}
                        />
                      )}
                    </div>
                    <div className="result-card-copy">
                      <strong>{breed.label}</strong>
                      <span>{breed.confidencePercent}% Match</span>
                    </div>
                  </div>
                ))
              ) : (
                <p className="empty-result">
                  {identifying
                    ? "Identifying animal..."
                    : "Take a photo to identify an animal."}
                </p>
              )}
            </div>

            {predictionSummary?.warning && (
              <IonText color="warning">
                <p className="warning-copy">{predictionSummary.warning}</p>
              </IonText>
            )}

            {highConfidenceCategory && (
              <div
                className={`category-result ${categoryMismatch ? "category-result-mismatch" : ""}`}
                role="status"
              >
                <strong>
                  {categoryMismatch
                    ? `This is a ${highConfidenceCategory.toLowerCase()}, not a ${category.toLowerCase()}.`
                    : `This is a ${highConfidenceCategory.toLowerCase()}.`}
                </strong>
                <span>
                  {predictionSummary?.confidencePercent.toFixed(1)}% confidence
                </span>
              </div>
            )}

            {breedExplanation.length > 0 && (
              <section
                className="breed-explanation"
                aria-labelledby="breed-explanation-title"
              >
                <h3 id="breed-explanation-title">Why these breeds?</h3>
                <ul>
                  {breedExplanation.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </section>
            )}
          </section>

          <div className="capture-actions">
            <input
              id="animal-camera"
              type="file"
              accept="image/*"
              capture="environment"
              hidden
              onChange={handlePhotoSelected}
            />
            <button
              className="capture-button"
              type="button"
              disabled={identifying}
              onClick={() => document.getElementById("animal-camera")?.click()}
              aria-label="Capture photo"
            >
              <span aria-hidden="true">⌾</span>
            </button>
          </div>

          {error && (
            <IonText color="danger">
              <p className="error-message">{error}</p>
            </IonText>
          )}
        </main>
      </IonContent>
    </IonPage>
  );
};

export default Home;
