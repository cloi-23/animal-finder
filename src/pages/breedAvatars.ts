import goldenRetrieverSvg from "../assets/breeds/golden-retriever.svg";
import poodleSvg from "../assets/breeds/poodle.svg";
import beagleSvg from "../assets/breeds/beagle.svg";
import shepherdSvg from "../assets/breeds/shepherd.svg";
import maineCoonSvg from "../assets/breeds/maine-coon.svg";
import persianSvg from "../assets/breeds/persian.svg";
import siameseSvg from "../assets/breeds/siamese.svg";
import bengalSvg from "../assets/breeds/bengal.svg";

export const getBreedAvatar = (label: string): string => {
  const value = label.toLowerCase();

  // Dogs
  if (value.includes("golden") || value.includes("retriever")) {
    return goldenRetrieverSvg;
  }
  if (value.includes("poodle")) {
    return poodleSvg;
  }
  if (value.includes("beagle")) {
    return beagleSvg;
  }
  if (value.includes("shepherd")) {
    return shepherdSvg;
  }
  if (value.includes("bulldog") || value.includes("boxer")) {
    return goldenRetrieverSvg;
  }
  if (value.includes("chihuahua") || value.includes("labrador")) {
    return goldenRetrieverSvg;
  }

  // Cats
  if (value.includes("maine coon")) {
    return maineCoonSvg;
  }
  if (value.includes("persian")) {
    return persianSvg;
  }
  if (value.includes("siamese")) {
    return siameseSvg;
  }
  if (value.includes("bengal")) {
    return bengalSvg;
  }
  if (value.includes("british shorthair") || value.includes("sphynx")) {
    return persianSvg;
  }
  if (value.includes("ragdoll") || value.includes("scottish fold")) {
    return siameseSvg;
  }

  // Default based on name
  if (
    value.includes("cat") ||
    value.includes("persian") ||
    value.includes("siamese")
  ) {
    return maineCoonSvg;
  }

  return goldenRetrieverSvg;
};
