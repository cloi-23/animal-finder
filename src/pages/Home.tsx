import { useEffect, useState } from "react";
import { AnimalAI } from "../ai/animalAi";
import {
  IonButton,
  IonContent,
  IonHeader,
  IonPage,
  IonText,
  IonTitle,
  IonToolbar,
} from "@ionic/react";
import { Animal, getAnimalByTaxonId } from "../database/animalDatabase";
import { useNavigate } from "react-router";
import { collectionService } from "../database/collectionService";
import { getBreedAvatar } from "./breedAvatars";
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
  const [prediction, setPrediction] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [identifiedAnimal, setIdentifiedAnimal] = useState<Animal | null>(null);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [identifying, setIdentifying] = useState(false);
  const [predictionSummary, setPredictionSummary] =
    useState<PredictionSummary | null>(null);

  useEffect(() => {
    AnimalAI.modelInfo().catch((modelError) =>
      console.error("Animal AI model unavailable", modelError),
    );
  }, []);

  const handlePhotoSelected = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setError("");
    setPrediction(null);
    setConfidence(null);
    setPredictionSummary(null);
    setIdentifiedAnimal(null);
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
      const result = await AnimalAI.classify({ image });
      console.log("AnimalAI classify result:", JSON.stringify(result));

      const summary = getPredictionSummary(result);
      setPredictionSummary(summary);
      setPrediction(
        summary.animal === "Unknown"
          ? "Unknown animal"
          : `${summary.animal} - ${summary.breed}`,
      );
      setConfidence(
        typeof result.confidence === "number" ? result.confidence : null,
      );

      // Track discovered breed in collection
      if (summary.isReliable && summary.breed !== "Unknown") {
        collectionService.recordBreedDiscovery(
          summary.breed,
          summary.animal as "Dog" | "Cat",
        );
      }

      if (summary.isReliable && result.taxonId !== null) {
        setIdentifiedAnimal(await getAnimalByTaxonId(result.taxonId));
      } else {
        setIdentifiedAnimal(null);
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
  const gi = topPredictions[0] ?? {
    label: "Golden Retriever",
    confidencePercent: 56.7,
  };
  const secondBreed = topPredictions[1] ?? {
    label: "Poodle",
    confidencePercent: 8.6,
  };
  const thirdBreed = topPredictions[2] ?? {
    label: "Other breeds less than 5%",
    confidencePercent: 34.7,
  };

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
              onClick={() => setCategory("Dog")}
            >
              🐕 Dogs
            </button>
            <button
              className={`category-btn ${category === "Cat" ? "active" : ""}`}
              onClick={() => setCategory("Cat")}
            >
              🐈 Cats
            </button>
          </div>

          <section className="scanner-panel">
            <h2>Your Result</h2>

            <div className="result-ring-wrap">
              <div
                className="result-ring"
                style={{
                  background: `conic-gradient(
                    #1ea6b4 0 ${gi.confidencePercent}%,
                    #1f5b66 ${gi.confidencePercent}% ${Math.min(
                      100,
                      gi.confidencePercent +
                        (secondBreed.confidencePercent || 8),
                    )}%,
                    #d8d8d8 ${Math.min(
                      100,
                      gi.confidencePercent +
                        (secondBreed.confidencePercent || 8),
                    )}% 100%
                  )`,
                }}
              >
                <div className="ring-center" />
              </div>

              {selectedImage ? (
                <div className="ring-avatar ring-avatar-main">
                  <img src={selectedImage} alt="Pet result" />
                </div>
              ) : (
                <div className="ring-avatar ring-avatar-main ring-avatar-placeholder">
                  <img
                    src={getBreedAvatar(topBreed.label)}
                    alt={topBreed.label}
                  />
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
              <div className="result-card result-card-primary">
                <div className="result-card-avatar">
                  {selectedImage ? (
                    <img src={selectedImage} alt="Top breed" />
                  ) : (
                    <img
                      src={getBreedAvatar(topBreed.label)}
                      alt={topBreed.label}
                    />
                  )}
                </div>
                <div className="result-card-copy">
                  <strong>{topBreed.label}</strong>
                  <span>{topBreed.confidencePercent}% Match</span>
                </div>
              </div>

              <div className="result-card">
                <div className="result-card-avatar">
                  <img
                    src={getBreedAvatar(secondBreed.label)}
                    alt={secondBreed.label}
                  />
                </div>
                <div className="result-card-copy">
                  <strong>{secondBreed.label}</strong>
                  <span>{secondBreed.confidencePercent}% Match</span>
                </div>
              </div>

              <div className="result-card">
                <div className="result-card-avatar">
                  <img
                    src={getBreedAvatar(thirdBreed.label)}
                    alt={thirdBreed.label}
                  />
                </div>
                <div className="result-card-copy">
                  <strong>{thirdBreed.label}</strong>
                  <span>{thirdBreed.confidencePercent}% Match</span>
                </div>
              </div>
            </div>

            {predictionSummary?.warning && (
              <IonText color="warning">
                <p className="warning-copy">{predictionSummary.warning}</p>
              </IonText>
            )}

            {identifiedAnimal && (
              <IonButton
                fill="outline"
                className="database-button"
                onClick={() => navigate(`/animal/${identifiedAnimal.id}`)}
              >
                Open database record
              </IonButton>
            )}
          </section>

          <div className="scanner-actions">
            <input
              id="animal-camera"
              type="file"
              accept="image/*"
              capture="environment"
              hidden
              onChange={handlePhotoSelected}
            />
            <input
              id="animal-upload"
              type="file"
              accept="image/*"
              hidden
              onChange={handlePhotoSelected}
            />

            <button className="tab tab-active" type="button">
              <span>◉</span>
              Scanner
            </button>
            <button
              className="tab"
              type="button"
              onClick={() => document.getElementById("animal-camera")?.click()}
            >
              <span>⏱</span>
            </button>
            <button
              className="tab"
              type="button"
              onClick={() => document.getElementById("animal-upload")?.click()}
            >
              <span>◌</span>
            </button>
            <button className="tab" type="button">
              <span>◍</span>
            </button>
            <button className="tab" type="button">
              <span>◔</span>
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
