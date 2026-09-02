export interface BreedRecord {
  breed: string;
  category: "Dog" | "Cat";
  discovered: boolean;
  firstScanDate?: string;
  scanCount: number;
  avatar?: string;
}

export interface CollectionData {
  breeds: Record<string, BreedRecord>;
  lastUpdated: string;
}

const STORAGE_KEY = "animal-finder-collection";

const DEFAULT_BREEDS = {
  dogs: [
    "Golden Retriever",
    "Poodle",
    "Beagle",
    "German Shepherd",
    "Bulldog",
    "Boxer",
    "Chihuahua",
    "Labrador Retriever",
  ],
  cats: [
    "Maine Coon",
    "Persian",
    "Siamese",
    "Bengal",
    "British Shorthair",
    "Sphynx",
    "Ragdoll",
    "Scottish Fold",
  ],
};

export const collectionService = {
  getCollection(): CollectionData {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        return JSON.parse(stored);
      } catch {
        return this.initializeCollection();
      }
    }
    return this.initializeCollection();
  },

  initializeCollection(): CollectionData {
    const breeds: Record<string, BreedRecord> = {};

    [...DEFAULT_BREEDS.dogs, ...DEFAULT_BREEDS.cats].forEach((breed) => {
      const category = DEFAULT_BREEDS.dogs.includes(breed) ? "Dog" : "Cat";
      breeds[breed] = {
        breed,
        category,
        discovered: false,
        scanCount: 0,
      };
    });

    const collection: CollectionData = {
      breeds,
      lastUpdated: new Date().toISOString(),
    };

    this.saveCollection(collection);
    return collection;
  },

  saveCollection(collection: CollectionData): void {
    collection.lastUpdated = new Date().toISOString();
    localStorage.setItem(STORAGE_KEY, JSON.stringify(collection));
  },

  recordBreedDiscovery(
    breed: string,
    category: "Dog" | "Cat",
    avatar?: string,
  ): void {
    const collection = this.getCollection();

    if (!collection.breeds[breed]) {
      collection.breeds[breed] = {
        breed,
        category,
        discovered: false,
        scanCount: 0,
      };
    }

    const record = collection.breeds[breed];
    if (!record.discovered) {
      record.discovered = true;
      record.firstScanDate = new Date().toISOString();
    }
    record.scanCount += 1;
    if (avatar) {
      record.avatar = avatar;
    }

    this.saveCollection(collection);
  },

  getDiscoveredBreeds(category?: "Dog" | "Cat"): BreedRecord[] {
    const collection = this.getCollection();
    return Object.values(collection.breeds)
      .filter(
        (record) =>
          record.discovered && (!category || record.category === category),
      )
      .sort((a, b) => {
        const dateA = new Date(a.firstScanDate || 0).getTime();
        const dateB = new Date(b.firstScanDate || 0).getTime();
        return dateB - dateA;
      });
  },

  getUndiscoveredBreeds(category?: "Dog" | "Cat"): BreedRecord[] {
    const collection = this.getCollection();
    return Object.values(collection.breeds)
      .filter(
        (record) =>
          !record.discovered && (!category || record.category === category),
      )
      .sort((a, b) => a.breed.localeCompare(b.breed));
  },

  getStats(category?: "Dog" | "Cat") {
    const collection = this.getCollection();
    const allBreeds = Object.values(collection.breeds).filter(
      (record) => !category || record.category === category,
    );
    const discovered = allBreeds.filter((r) => r.discovered).length;

    return {
      total: allBreeds.length,
      discovered,
      remaining: allBreeds.length - discovered,
      discoveryPercentage: allBreeds.length
        ? (discovered / allBreeds.length) * 100
        : 0,
    };
  },

  clearCollection(): void {
    localStorage.removeItem(STORAGE_KEY);
  },
};
