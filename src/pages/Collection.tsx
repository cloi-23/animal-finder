import { useEffect, useState } from "react";
import { IonContent, IonPage, IonText, IonButton } from "@ionic/react";
import { useNavigate } from "react-router";
import {
  collectionService,
  type BreedRecord,
} from "../database/collectionService";
import { getBreedAvatar } from "./breedAvatars";
import "./Collection.css";

const Collection: React.FC = () => {
  const navigate = useNavigate();
  const [category, setCategory] = useState<"Dog" | "Cat">("Dog");
  const [discovered, setDiscovered] = useState<BreedRecord[]>([]);
  const [undiscovered, setUndiscovered] = useState<BreedRecord[]>([]);
  const [stats, setStats] = useState({
    total: 0,
    discovered: 0,
    remaining: 0,
    discoveryPercentage: 0,
  });

  useEffect(() => {
    loadCollection();
  }, [category]);

  const loadCollection = () => {
    setDiscovered(collectionService.getDiscoveredBreeds(category));
    setUndiscovered(collectionService.getUndiscoveredBreeds(category));
    setStats(collectionService.getStats(category));
  };

  return (
    <IonPage>
      <IonContent fullscreen>
        <main className="collection-home">
          <header className="collection-header">
            <h1>COLLECTION</h1>
            <IonButton
              fill="clear"
              size="small"
              className="back-button"
              onClick={() => navigate("/")}
            >
              ✕
            </IonButton>
          </header>

          <section className="collection-panel">
            <div className="category-tabs">
              <button
                className={`category-tab ${category === "Dog" ? "active" : ""}`}
                onClick={() => setCategory("Dog")}
              >
                🐕 Dogs
              </button>
              <button
                className={`category-tab ${category === "Cat" ? "active" : ""}`}
                onClick={() => setCategory("Cat")}
              >
                🐈 Cats
              </button>
            </div>

            <div className="collection-stats">
              <div className="stat-box">
                <span className="stat-label">DISCOVERED</span>
                <span className="stat-value">{stats.discovered}</span>
                <span className="stat-subtitle">of {stats.total}</span>
              </div>
              <div className="stat-progress">
                <div
                  className="progress-bar"
                  style={{
                    width: `${stats.discoveryPercentage}%`,
                  }}
                />
                <span className="progress-text">
                  {stats.discoveryPercentage.toFixed(0)}%
                </span>
              </div>
            </div>

            {discovered.length > 0 && (
              <div className="breeds-section">
                <h2>✓ Discovered</h2>
                <div className="breeds-grid">
                  {discovered.map((record) => (
                    <div key={record.breed} className="breed-card discovered">
                      <div className="breed-avatar">
                        <img
                          src={getBreedAvatar(record.breed)}
                          alt={record.breed}
                        />
                      </div>
                      <div className="breed-info">
                        <strong>{record.breed}</strong>
                        <span className="scan-count">{record.scanCount}x</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {undiscovered.length > 0 && (
              <div className="breeds-section">
                <h2>? Unknown</h2>
                <div className="breeds-grid">
                  {undiscovered.map((record) => (
                    <div key={record.breed} className="breed-card undiscovered">
                      <div className="breed-avatar locked">
                        <span>🔒</span>
                      </div>
                      <div className="breed-info">
                        <strong>{record.breed}</strong>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>

          <IonButton
            fill="clear"
            className="scan-more-button"
            onClick={() => navigate("/")}
          >
            ← Back to Scanner
          </IonButton>
        </main>
      </IonContent>
    </IonPage>
  );
};

export default Collection;
