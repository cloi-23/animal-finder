import { useEffect, useState } from "react";
import { AnimalAI } from "../ai/animalAi";
import {
  IonContent,
  IonHeader,
  IonPage,
  IonTitle,
  IonToolbar,
  IonSearchbar,
  IonCard,
  IonCardContent,
  IonText,
  IonSpinner,
  IonButton,
} from "@ionic/react";

import {
  searchAnimals,
  Animal,
  getDistribution,
  getAnimalByTaxonId,
} from "../database/animalDatabase";

import "./Home.css";
import { useNavigate } from "react-router";

const Home: React.FC = () => {
  const navigate = useNavigate();

  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Animal[]>([]);
  const [distribution, setDistribution] = useState<Record<number, string[]>>(
    {},
  );

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [prediction, setPrediction] = useState<string | null>(null);
  const [predictionConfidence, setPredictionConfidence] = useState<
    number | null
  >(null);
  const [identifiedAnimal, setIdentifiedAnimal] = useState<Animal | null>(null);

  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [identifying, setIdentifying] = useState(false);

  useEffect(() => {
    AnimalAI.modelInfo()
      .then((info) => {
        console.log("ANIMAL AI MODEL INFO", JSON.stringify(info, null, 2));
      })
      .catch((error) => {
        console.error("ANIMAL AI MODEL INFO FAILED", error);
      });
  }, []);

  const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();

      reader.onload = () => {
        if (typeof reader.result !== "string") {
          reject(new Error("Unable to convert image to Base64"));
          return;
        }

        resolve(reader.result);
      };

      reader.onerror = () => {
        reject(reader.error ?? new Error("Unable to read image"));
      };

      reader.readAsDataURL(file);
    });
  };

  const handlePhotoSelected = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];

    if (!file) return;

    setError("");
    setPrediction(null);
    setPredictionConfidence(null);
    setIdentifiedAnimal(null);
    setIdentifying(true);

    const imageUrl = URL.createObjectURL(file);
    setSelectedImage(imageUrl);

    try {
      console.log("Animal AI: converting image to Base64...");

      const base64Image = await fileToBase64(file);

      console.log("Animal AI: image converted");
      console.log("Animal AI: Base64 length:", base64Image.length);

      console.log("Animal AI: running classification...");

      const result = await AnimalAI.classify({
        image: base64Image,
      });

      console.log("Animal AI RESULT:", JSON.stringify(result, null, 2));
      setPrediction(`${result.category} — ${result.name}`);
      if (result.taxonId !== null) {
        setIdentifiedAnimal(await getAnimalByTaxonId(result.taxonId));
      }
      setPredictionConfidence(
        typeof result.confidence === "number" ? result.confidence : null,
      );
    } catch (err) {
      console.error("Animal AI classification failed:", err);

      setError(
        err instanceof Error ? err.message : "Unable to identify the animal.",
      );
    } finally {
      setIdentifying(false);

      /*
       * Allow selecting the same file again.
       */
      event.target.value = "";
    }
  };

  const handleSearch = async (value: string) => {
    setQuery(value);
    setError("");

    if (!value.trim()) {
      setResults([]);
      setDistribution({});
      return;
    }

    setLoading(true);

    try {
      const animals = await searchAnimals(value);
      setResults(animals);
    } catch (err) {
      console.error(err);
      setError("Unable to search the animal database.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let cancelled = false;

    const loadDistribution = async () => {
      if (results.length === 0) {
        setDistribution({});
        return;
      }

      const entries = await Promise.all(
        results.map(async (animal) => {
          try {
            const areas = await getDistribution(animal.id);
            return [animal.id, areas] as const;
          } catch (err) {
            console.error(
              `Unable to load distribution for ${animal.scientific_name}`,
              err,
            );

            return [animal.id, []] as const;
          }
        }),
      );

      if (!cancelled) {
        setDistribution(Object.fromEntries(entries));
      }
    };

    loadDistribution();

    return () => {
      cancelled = true;
    };
  }, [results]);

  return (
    <IonPage>
      <IonHeader>
        <IonToolbar>
          <IonTitle>🐾 Animal Finder</IonTitle>
        </IonToolbar>
      </IonHeader>

      <IonContent fullscreen>
        <div className="animal-home">
          <IonText>
            <h1>Animal Finder</h1>

            <p>
              Search millions of animals from the offline Catalogue of Life
              database.
            </p>
          </IonText>

          <IonSearchbar
            value={query}
            placeholder="Search animals..."
            debounce={500}
            onIonInput={(event) => handleSearch(event.detail.value ?? "")}
          />

          <div className="photo-identify">
            <input
              id="animal-photo"
              type="file"
              accept="image/*"
              capture="environment"
              hidden
              onChange={handlePhotoSelected}
            />

            <IonButton
              expand="block"
              size="large"
              disabled={identifying}
              onClick={() => {
                document.getElementById("animal-photo")?.click();
              }}
            >
              {identifying ? "🧠 Identifying..." : "📷 Identify from photo"}
            </IonButton>

            <p>
              {identifying
                ? "The AI model is analyzing the image..."
                : "Take a photo or choose one from your device."}
            </p>

            {selectedImage && (
              <div className="photo-preview">
                <img src={selectedImage} alt="Selected animal" />
              </div>
            )}
          </div>

          {prediction && (
            <div className="prediction">
              <h2>🧠 Identification</h2>

              <h3>{prediction}</h3>

              {predictionConfidence !== null && (
                <p>Confidence: {Math.round(predictionConfidence * 100)}%</p>
              )}

              {identifiedAnimal && (
                <IonButton
                  fill="outline"
                  onClick={() => navigate(`/animal/${identifiedAnimal.id}`)}
                >
                  Open database record
                </IonButton>
              )}
            </div>
          )}

          {loading && (
            <div className="loading">
              <IonSpinner />
              <p>Searching database...</p>
            </div>
          )}

          {error && (
            <IonText color="danger">
              <p>{error}</p>
            </IonText>
          )}

          {!loading && results.length > 0 && (
            <div>
              <IonText>
                <p>{results.length} result(s)</p>
              </IonText>

              {results.map((animal) => {
                const areas = distribution[animal.id] ?? [];

                return (
                  <IonCard
                    key={animal.id}
                    button
                    onClick={() => navigate(`/animal/${animal.id}`)}
                  >
                    <IonCardContent>
                      {animal.common_name && <h2>{animal.common_name}</h2>}

                      <h3>{animal.scientific_name}</h3>

                      {animal.authorship && (
                        <p className="authorship">{animal.authorship}</p>
                      )}

                      <div className="taxonomy">
                        {animal.kingdom && <span>{animal.kingdom}</span>}

                        {animal.phylum && <span>{animal.phylum}</span>}

                        {animal.class_name && <span>{animal.class_name}</span>}

                        {animal.order_name && <span>{animal.order_name}</span>}

                        {animal.family && <span>{animal.family}</span>}

                        {animal.genus && <span>{animal.genus}</span>}
                      </div>

                      {areas.length > 0 && (
                        <div className="distribution">
                          <h4>🌍 Distribution</h4>

                          <ul>
                            {areas.slice(0, 10).map((area) => (
                              <li key={area}>{area}</li>
                            ))}
                          </ul>

                          {areas.length > 10 && (
                            <p>+ {areas.length - 10} more areas</p>
                          )}
                        </div>
                      )}

                      {animal.extinct === 1 && (
                        <IonText color="danger">
                          <p>Extinct</p>
                        </IonText>
                      )}
                    </IonCardContent>
                  </IonCard>
                );
              })}
            </div>
          )}

          {!loading && query.trim() && results.length === 0 && !error && (
            <div className="no-results">
              <h2>No animals found</h2>
              <p>Try another name.</p>
            </div>
          )}
        </div>
      </IonContent>
    </IonPage>
  );
};

export default Home;
