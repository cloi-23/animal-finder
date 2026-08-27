import { useEffect, useState } from "react";
import {
  IonBackButton,
  IonButtons,
  IonCard,
  IonCardContent,
  IonContent,
  IonHeader,
  IonPage,
  IonSpinner,
  IonTitle,
  IonToolbar,
  IonText,
} from "@ionic/react";
import { useParams } from "react-router-dom";

import {
  Animal,
  getAnimalById,
  getDistribution,
} from "../database/animalDatabase";

import "./AnimalDetails.css";

const AnimalDetails: React.FC = () => {
  const { id } = useParams<{ id: string }>();

  const [animal, setAnimal] = useState<Animal | null>(null);
  const [distribution, setDistribution] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        const animalId = Number(id);

        const found = await getAnimalById(animalId);

        if (!found) {
          setLoading(false);
          return;
        }

        setAnimal(found);

        const areas = await getDistribution(animalId);
        setDistribution(areas);
      } catch (error) {
        console.error("Unable to load animal:", error);
      } finally {
        setLoading(false);
      }
    };

    load();
  }, [id]);

  if (loading) {
    return (
      <IonPage>
        <IonContent className="ion-padding">
          <IonSpinner />
          <p>Loading animal...</p>
        </IonContent>
      </IonPage>
    );
  }

  if (!animal) {
    return (
      <IonPage>
        <IonHeader>
          <IonToolbar>
            <IonButtons slot="start">
              <IonBackButton defaultHref="/home" />
            </IonButtons>
            <IonTitle>Animal</IonTitle>
          </IonToolbar>
        </IonHeader>

        <IonContent className="ion-padding">
          <h2>Animal not found</h2>
        </IonContent>
      </IonPage>
    );
  }

  return (
    <IonPage>
      <IonHeader>
        <IonToolbar>
          <IonButtons slot="start">
            <IonBackButton defaultHref="/home" />
          </IonButtons>

          <IonTitle>{animal.common_name || animal.scientific_name}</IonTitle>
        </IonToolbar>
      </IonHeader>

      <IonContent fullscreen>
        <div className="animal-details">
          <IonCard>
            <IonCardContent>
              {animal.common_name && <h1>{animal.common_name}</h1>}

              <h2>{animal.scientific_name}</h2>

              {animal.authorship && (
                <p className="authorship">{animal.authorship}</p>
              )}

              {animal.status && (
                <p>
                  <strong>Status:</strong> {animal.status}
                </p>
              )}

              {animal.extinct === 1 && (
                <IonText color="danger">
                  <p>
                    <strong>Extinct</strong>
                  </p>
                </IonText>
              )}
            </IonCardContent>
          </IonCard>

          <IonCard>
            <IonCardContent>
              <h2>Classification</h2>

              <div className="classification">
                {animal.kingdom && (
                  <div>
                    <strong>Kingdom</strong>
                    <span>{animal.kingdom}</span>
                  </div>
                )}

                {animal.phylum && (
                  <div>
                    <strong>Phylum</strong>
                    <span>{animal.phylum}</span>
                  </div>
                )}

                {animal.class_name && (
                  <div>
                    <strong>Class</strong>
                    <span>{animal.class_name}</span>
                  </div>
                )}

                {animal.order_name && (
                  <div>
                    <strong>Order</strong>
                    <span>{animal.order_name}</span>
                  </div>
                )}

                {animal.family && (
                  <div>
                    <strong>Family</strong>
                    <span>{animal.family}</span>
                  </div>
                )}

                {animal.genus && (
                  <div>
                    <strong>Genus</strong>
                    <span>{animal.genus}</span>
                  </div>
                )}
              </div>
            </IonCardContent>
          </IonCard>

          {distribution.length > 0 && (
            <IonCard>
              <IonCardContent>
                <h2>🌍 Distribution</h2>

                <ul>
                  {distribution.map((area) => (
                    <li key={area}>{area}</li>
                  ))}
                </ul>
              </IonCardContent>
            </IonCard>
          )}
        </div>
      </IonContent>
    </IonPage>
  );
};

export default AnimalDetails;
