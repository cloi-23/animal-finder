import initSqlJs, { Database } from "sql.js";

let db: Database | null = null;
let loading: Promise<Database> | null = null;

export interface Animal {
  id: number;
  scientific_name: string;
  authorship: string | null;
  status: string | null;

  common_name: string | null;

  kingdom: string | null;
  phylum: string | null;
  class_name: string | null;
  order_name: string | null;
  family: string | null;
  genus: string | null;

  extinct: number;
}

async function loadDatabase(): Promise<Database> {
  if (db) return db;

  if (loading) return loading;

  loading = (async () => {
    const SQL = await initSqlJs({
      locateFile: () => "/sql-wasm.wasm",
    });

    const response = await fetch("/database/animals_mobile.sqlite");

    if (!response.ok) {
      throw new Error(`Could not load animal database: ${response.status}`);
    }

    const buffer = await response.arrayBuffer();

    console.log(
      `Animal database loaded: ${(buffer.byteLength / 1024 / 1024).toFixed(1)} MB`,
    );

    db = new SQL.Database(new Uint8Array(buffer));

    return db;
  })();

  return loading;
}

export async function searchAnimals(
  searchTerm: string,
  limit = 30,
): Promise<Animal[]> {
  const database = await loadDatabase();

  const term = searchTerm.trim();

  if (!term) return [];

  const pattern = `%${term}%`;

  const result = database.exec(
    `
    SELECT DISTINCT
      s.id,
      s.scientific_name,
      s.authorship,
      s.status,
      
      cn.name AS common_name,
      k.name AS kingdom,
      p.name AS phylum,
      c.name AS class_name,
      o.name AS order_name,
      f.name AS family,
      g.name AS genus,

      s.extinct

    FROM species s

    LEFT JOIN taxa k ON k.id = s.kingdom_id
    LEFT JOIN taxa p ON p.id = s.phylum_id
    LEFT JOIN taxa c ON c.id = s.class_id
    LEFT JOIN taxa o ON o.id = s.order_id
    LEFT JOIN taxa f ON f.id = s.family_id
    LEFT JOIN taxa g ON g.id = s.genus_id

    LEFT JOIN common_names cn
      ON cn.species_id = s.id

    WHERE
      s.scientific_name LIKE $pattern
      OR cn.name LIKE $pattern

    ORDER BY
      CASE
        WHEN LOWER(s.scientific_name) = LOWER($term) THEN 0
        WHEN LOWER(cn.name) = LOWER($term) THEN 1
        WHEN LOWER(s.scientific_name) LIKE LOWER($prefix) THEN 2
        WHEN LOWER(cn.name) LIKE LOWER($prefix) THEN 3
        ELSE 4
      END,
      s.scientific_name

    LIMIT $limit
    `,
    {
      $pattern: pattern,
      $term: term,
      $prefix: `${term}%`,
      $limit: limit,
    },
  );

  if (!result.length) return [];

  return result[0].values.map((row: unknown[]) => ({
    id: row[0] as number,
    scientific_name: row[1] as string,
    authorship: row[2] as string | null,
    status: row[3] as string | null,

    common_name: row[4] as string | null,
    kingdom: row[5] as string | null,
    phylum: row[6] as string | null,
    class_name: row[7] as string | null,
    order_name: row[8] as string | null,
    family: row[9] as string | null,
    genus: row[10] as string | null,

    extinct: Number(row[11] ?? 0),
  }));
}

export async function getDistribution(speciesId: number): Promise<string[]> {
  const database = await loadDatabase();

  const result = database.exec(
    `
    SELECT DISTINCT area
    FROM distribution
    WHERE species_id = $speciesId
      AND area IS NOT NULL
      AND TRIM(area) != ''
    ORDER BY area
    `,
    {
      $speciesId: speciesId,
    },
  );

  if (!result.length) return [];

  return result[0].values
    .map((row: unknown[]) => String(row[0]))
    .filter(Boolean);
}

export async function getAnimalById(id: number): Promise<Animal | null> {
  const database = await loadDatabase();

  const result = database.exec(
    `
    SELECT
      s.id,
      s.scientific_name,
      s.authorship,
      s.status,

      (
        SELECT cn.name
        FROM common_names cn
        WHERE cn.species_id = s.id
        ORDER BY cn.preferred DESC, cn.name
        LIMIT 1
      ) AS common_name,

      k.name AS kingdom,
      p.name AS phylum,
      c.name AS class_name,
      o.name AS order_name,
      f.name AS family,
      g.name AS genus,

      s.extinct

    FROM species s

    LEFT JOIN taxa k ON k.id = s.kingdom_id
    LEFT JOIN taxa p ON p.id = s.phylum_id
    LEFT JOIN taxa c ON c.id = s.class_id
    LEFT JOIN taxa o ON o.id = s.order_id
    LEFT JOIN taxa f ON f.id = s.family_id
    LEFT JOIN taxa g ON g.id = s.genus_id

    WHERE s.id = $id
    LIMIT 1
    `,
    { $id: id },
  );

  if (!result.length || !result[0].values.length) {
    return null;
  }

  const row = result[0].values[0] as unknown[];

  return {
    id: row[0] as number,
    scientific_name: row[1] as string,
    authorship: row[2] as string | null,
    status: row[3] as string | null,
    common_name: row[4] as string | null,

    kingdom: row[5] as string | null,
    phylum: row[6] as string | null,
    class_name: row[7] as string | null,
    order_name: row[8] as string | null,
    family: row[9] as string | null,
    genus: row[10] as string | null,

    extinct: Number(row[11] ?? 0),
  };
}
