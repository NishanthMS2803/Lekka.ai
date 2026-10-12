#!/usr/bin/env node
import { execSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const frontendDir = path.resolve(__dirname, "..");
const repoRoot = path.resolve(frontendDir, "..");
const openapiJsonPath = path.resolve(frontendDir, "openapi.json");
const schemaTsPath = path.resolve(frontendDir, "lib/api/schema.d.ts");

console.log("Generating OpenAPI schema and TypeScript definitions...");

let schemaExtracted = false;

// 1. Try dumping schema directly via python from the Lekka AI backend
try {
  console.log("Extracting OpenAPI schema from FastAPI backend...");
  const dummyEnv = {
    ...process.env,
    DATABASE_URL: process.env.DATABASE_URL || "sqlite:///dummy.db",
    GEMINI_API_KEY: process.env.GEMINI_API_KEY || "dummy_api_key",
    GEMINI_MODEL: process.env.GEMINI_MODEL || "gemini-2.5-flash",
    GEMINI_EMBEDDING_MODEL: process.env.GEMINI_EMBEDDING_MODEL || "text-embedding-004",
    SEC_USER_AGENT: process.env.SEC_USER_AGENT || "Lekka test@lekka.ai",
  };

  const schemaJson = execSync(
    'uv run python -c "import json; from lekka_ai.api.app import create_app; print(json.dumps(create_app().openapi(), indent=2))"',
    {
      cwd: repoRoot,
      env: dummyEnv,
      encoding: "utf-8",
      maxBuffer: 10 * 1024 * 1024,
    }
  );

  if (schemaJson && schemaJson.trim().startsWith("{")) {
    fs.writeFileSync(openapiJsonPath, schemaJson.trim(), "utf-8");
    schemaExtracted = true;
    console.log(`Saved updated schema to ${openapiJsonPath}`);
  }
} catch (err) {
  console.warn("Could not dump schema via python:", err.message);
}

// 2. Fallback check: if python extraction didn't happen, ensure openapi.json exists
if (!schemaExtracted) {
  if (fs.existsSync(openapiJsonPath)) {
    console.log(`Using existing ${openapiJsonPath}`);
  } else {
    console.error("Error: openapi.json not found and backend extraction failed.");
    process.exit(1);
  }
}

// 3. Ensure destination directory exists
fs.mkdirSync(path.dirname(schemaTsPath), { recursive: true });

// 4. Run openapi-typescript to generate schema.d.ts
try {
  console.log("Running openapi-typescript...");
  execSync(`npx openapi-typescript "${openapiJsonPath}" -o "${schemaTsPath}"`, {
    cwd: frontendDir,
    stdio: "inherit",
  });
  console.log(`Successfully generated TypeScript types at ${schemaTsPath}`);
} catch (err) {
  console.error("Failed to generate TypeScript types:", err);
  process.exit(1);
}
