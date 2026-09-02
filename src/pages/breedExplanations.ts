export type ExplanationPrediction = {
  label: string;
  confidencePercent: number;
};

const CAT_EXPLANATIONS: Record<string, string> = {
  Abyssinian:
    "Usually has a short, close coat with a distinctive ticked pattern, large ears, and an athletic body. A fluffy white kitten would be a poor visual match.",
  Bengal:
    "Usually has a short coat with distinctive spots or rosettes, sometimes with a marbled pattern. A long, fluffy white coat does not resemble the typical Bengal appearance.",
  Birman:
    "Usually has a medium-to-long silky coat, darker coloring on the ears, face, legs, and tail, and characteristic white gloves on the paws. An entirely white kitten does not strongly match.",
  Bombay:
    "Usually has a sleek, short, completely black coat, with a rounded head and copper or gold eyes. A white fluffy kitten does not match the typical appearance.",
  "British Shorthair":
    "Usually has a dense, short plush coat, a broad rounded head, and a stocky body. A long fluffy coat makes this less likely.",
  "Egyptian Mau":
    "Usually has a short coat with clearly visible natural spots and a lean, athletic build. A fluffy white coat is not typical.",
  "Maine Coon":
    "Usually has large ears with prominent tufts, a long muzzle, large paws, and a long shaggy coat. Some traits could overlap with a fluffy kitten, but the distinctive features should be obvious.",
  Persian:
    "Usually has a very dense, long coat and a rounded head; many have a noticeably short, flat nose. A relatively normal-shaped face makes this possible but less strongly supported.",
  Ragdoll:
    "Usually has a semi-long coat, blue eyes, and darker point coloring on the face, ears, legs, and tail. A completely white kitten does not strongly match the typical pattern.",
  "Russian Blue":
    "Usually has a short, dense blue-gray coat, green eyes, and a slender build. A white, long-haired appearance does not match.",
  Siamese:
    "Usually has a short coat with dark color points on the face, ears, legs, and tail, along with a slender body and blue eyes. A fluffy white coat does not match well.",
  Sphynx:
    "Characteristically has very little visible hair, large ears, and wrinkled skin. This does not match a fluffy kitten.",
};

const DOG_EXPLANATIONS: Record<string, string> = {
  "American Bulldog":
    "Usually has a muscular body, broad head, short coat, and strong muzzle. A fluffy kitten would not visually match.",
  "American Pit Bull Terrier":
    "Usually has a muscular build, broad head, short coat, and relatively short hair. A fluffy kitten does not match.",
  "Basset Hound":
    "Usually has very long floppy ears, short legs, loose skin, and a long body. None of those features are visible here.",
  Beagle:
    "Usually has a short coat, long floppy ears, and characteristic hound coloring. A fluffy white kitten does not resemble a Beagle.",
  Boxer:
    "Usually has a short coat, muscular body, square muzzle, and broad head. A kitten clearly does not match.",
  Chihuahua:
    "Usually is very small with a relatively large head and large upright ears, but has a short or smooth coat in many individuals. A fluffy kitten's feline facial features do not match.",
  "English Cocker Spaniel":
    "Usually has a medium-length silky coat and long floppy ears. The ears, face, and body shape do not match.",
  "English Setter":
    "Usually has a medium-to-long feathered coat, long muzzle, and hanging ears. The body and facial structure do not resemble this kitten.",
  "German Shorthaired":
    "Usually has a short coat, long muzzle, floppy ears, and an athletic build. A fluffy white kitten is not a visual match.",
  "Great Pyrenees":
    "Usually is a very large dog with a thick white double coat and a broad head. The white fluffy coat could superficially overlap, but the face and body proportions are feline.",
  Havanese:
    "Usually is a small dog with a long, soft, often wavy coat and floppy ears. The facial structure and posture do not match.",
  "Japanese Chin":
    "Usually is a small toy dog with a broad face, large eyes, short muzzle, and long silky coat. Despite the fluffy appearance, the facial structure is different.",
  Keeshond:
    "Usually has a thick double coat with distinctive gray, black, and cream coloring and a fox-like face. A completely white coat does not match well.",
  Leonberger:
    "Usually is a very large dog with a long coat, broad muzzle, and substantial body. The size and facial structure are completely different.",
  "Miniature Pinscher":
    "Usually has a short, smooth coat, slender muscular body, and upright ears. A fluffy white kitten is not a good match.",
  Newfoundland:
    "Usually is a very large, heavily built dog with a thick water-resistant coat. The proportions and face do not match a kitten.",
  Pomeranian:
    "Usually is a very small dog with an extremely fluffy double coat, pointed ears, and a fox-like face. The fluffy coat could cause confusion, but the facial structure is different.",
  Pug: "Usually has a very short muzzle, rounded head, prominent eyes, and a short coat. A kitten does not match those traits.",
  "Saint Bernard":
    "Usually is a very large dog with a broad head, large body, and short or long coat. The proportions and face do not match a kitten.",
  Samoyed:
    "Usually has a thick, fluffy white coat, upright triangular ears, and a broad dog-like muzzle. The white fluffy coat could explain confusion, but the face and body are feline.",
  "Scottish Terrier":
    "Usually has a wiry coat, distinctive beard, long muzzle, and short legs. A kitten does not resemble it.",
  "Shiba Inu":
    "Usually has a compact body, upright ears, curled tail, and short dense coat. A fluffy white kitten does not match.",
  "Staffordshire Bull Terrier":
    "Usually has a muscular body, broad head, short coat, and strong muzzle. A kitten does not match.",
  "Wheaten Terrier":
    "Usually has a soft, wavy, wheaten-colored coat and a distinctive beard. The white coat and feline facial structure do not match.",
  "Yorkshire Terrier":
    "Usually has a long, fine, straight coat that is typically blue-gray and tan, along with small upright ears. A white fluffy kitten does not match.",
};

export function getBreedExplanation(
  category: string,
  predictions: ExplanationPrediction[],
): string[] {
  if (predictions.length === 0 || (category !== "Cat" && category !== "Dog")) {
    return [];
  }

  const explanations = category === "Cat" ? CAT_EXPLANATIONS : DOG_EXPLANATIONS;
  const items = predictions
    .slice(0, 3)
    .filter((prediction) => explanations[prediction.label])
    .map(
      (prediction) =>
        `${prediction.label} (${prediction.confidencePercent}%): ${explanations[prediction.label]}`,
    );

  if (category === "Cat" && predictions[0]?.label === "Bengal") {
    items.push(
      "Other possibility: Domestic Longhair is a more plausible description for a white, fluffy cat without distinctive pedigree traits. A Turkish Angora-type cat could also be considered.",
    );
    if (predictions[0].confidencePercent < 50) {
      items.push(
        `Conclusion: The model's ${predictions[0].confidencePercent.toFixed(2)}% Bengal prediction is low confidence. The safest result is Cat - Breed Unknown.`,
      );
    }
  }

  return items;
}
