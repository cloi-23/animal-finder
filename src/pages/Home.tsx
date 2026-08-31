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
import "./Home.css";

const categories = [
  { icon: "🐕", label: "Dogs", color: "category-dog" },
  { icon: "🐈", label: "Cats", color: "category-cat" },
  { icon: "🦜", label: "Birds", color: "category-bird" },
  { icon: "🐠", label: "Fish", color: "category-fish" },
];

const Home: React.FC = () => {
  const navigate = useNavigate();
  const [error, setError] = useState("");
  const [prediction, setPrediction] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [identifiedAnimal, setIdentifiedAnimal] = useState<Animal | null>(null);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [identifying, setIdentifying] = useState(false);

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
      setPrediction(`${result.category} - ${result.name}`);
      setConfidence(
        typeof result.confidence === "number" ? result.confidence : null,
      );
      if (result.taxonId !== null)
        setIdentifiedAnimal(await getAnimalByTaxonId(result.taxonId));
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

  return (
    <IonPage>
      <IonHeader>
        <IonToolbar>
          <IonTitle>Animal Finder</IonTitle>
        </IonToolbar>
      </IonHeader>
      <IonContent fullscreen>
        <main className="animal-home">
          <section className="welcome-copy">
            <p className="eyebrow">SMART PET</p>
            <h1>
              Happy pets,
              <br />
              happy you.
            </h1>
            <p>Discover your pet's breed with one photo.</p>
          </section>

          <section className="category-section" aria-label="Animal categories">
            <div className="category-heading">
              <h2>Categories</h2>
              <span>Explore</span>
            </div>
            <div className="category-row">
              {categories.map((category) => (
                <div
                  className={`category-item ${category.color}`}
                  key={category.label}
                >
                  <span>{category.icon}</span>
                  <small>{category.label}</small>
                </div>
              ))}
            </div>
          </section>

          <section className="photo-identify">
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
            <div className="photo-copy">
              <span className="camera-mark">✦</span>
              <h2>Meet your pet's match</h2>
              <p>
                {identifying
                  ? "Analyzing your photo..."
                  : "Take a photo and let AI identify the breed."}
              </p>
            </div>
            <div className="photo-actions">
              <IonButton
                size="large"
                disabled={identifying}
                onClick={() =>
                  document.getElementById("animal-camera")?.click()
                }
              >
                {identifying ? "Identifying..." : "Take a picture"}
              </IonButton>
              <IonButton
                className="upload-button"
                fill="outline"
                size="large"
                disabled={identifying}
                onClick={() =>
                  document.getElementById("animal-upload")?.click()
                }
              >
                Upload a photo
              </IonButton>
            </div>
            {selectedImage && (
              <div className="photo-preview">
                <img src={selectedImage} alt="Selected animal" />
              </div>
            )}
          </section>

          {prediction && (
            <section className="prediction">
              <p className="eyebrow">IDENTIFIED PET</p>
              <h2>{prediction}</h2>
              {confidence !== null && (
                <p>Confidence: {Math.round(confidence * 100)}%</p>
              )}
              {identifiedAnimal && (
                <IonButton
                  fill="outline"
                  onClick={() => navigate(`/animal/${identifiedAnimal.id}`)}
                >
                  Open database record
                </IonButton>
              )}
            </section>
          )}
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
