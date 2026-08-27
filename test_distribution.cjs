const fs = require("fs");
const initSqlJs = require("sql.js");

(async () => {
  const SQL = await initSqlJs({
    locateFile: file => `node_modules/sql.js/dist/${file}`
  });

  const buffer = fs.readFileSync("dist/database/animals_mobile.sqlite");
  const db = new SQL.Database(buffer);

  const result = db.exec(`
    SELECT DISTINCT
      d.area
    FROM distribution d
    JOIN species s ON s.id = d.species_id
    WHERE s.scientific_name = 'Panthera leo'
      AND d.area IS NOT NULL
      AND TRIM(d.area) != ''
    ORDER BY d.area
  `);

  console.table(result[0]?.values || []);

  db.close();
})();
