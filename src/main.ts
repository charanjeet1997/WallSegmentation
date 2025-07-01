import * as ort from "onnxruntime-web";
import { env } from "onnxruntime-web";

const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const ctx = canvas.getContext("2d")!;
const imageInput = document.getElementById("imageInput") as HTMLInputElement;

env.wasm.wasmPaths = "/ort/";

async function loadModel() {
  console.log("Loading ONNX model...");
  if (!ort.env.wasm) {
    console.error("WASM execution provider is not available.");
    return null;
  }

  try {
    const session = await ort.InferenceSession.create("../ONNX/segformer.onnx", {
      executionProviders: ["wasm"],
    });
    console.log("Model loaded successfully", session);
    return session;
  } catch (error) {
    console.error("Error loading model:", error);
    throw error;
  }
}

function preprocessImage(image: HTMLImageElement): Float32Array {
  const width = 512;
  const height = 512;
  canvas.width = width;
  canvas.height = height;

  ctx.drawImage(image, 0, 0, width, height);
  const { data } = ctx.getImageData(0, 0, width, height);

  const floatData = new Float32Array(3 * width * height);
  for (let i = 0; i < width * height; i++) {
    floatData[i] = data[i * 4] / 255;
    floatData[i + width * height] = data[i * 4 + 1] / 255;
    floatData[i + 2 * width * height] = data[i * 4 + 2] / 255;
  }

  return floatData;
}

function loadTexture(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.src = src;
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = reject;
  });
}

function createTiledPattern(texture: HTMLImageElement, scale: number): HTMLCanvasElement {
  const tileWidth = texture.width * scale;
  const tileHeight = texture.height * scale;

  const patternCanvas = document.createElement("canvas");
  patternCanvas.width = tileWidth;
  patternCanvas.height = tileHeight;

  const patternCtx = patternCanvas.getContext("2d")!;
  patternCtx.drawImage(texture, 0, 0, tileWidth, tileHeight);

  return patternCanvas;
}

// Optional mask expansion to clean small gaps
function dilate(mask: Uint8Array, width: number, height: number): Uint8Array {
  const result = new Uint8Array(mask.length);
  const radius = 1;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (mask[idx] === 0) continue;

      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            result[ny * width + nx] = 1;
          }
        }
      }
    }
  }

  return result;
}

async function drawSegmentationOverlay(
    outputClasses: Uint8Array,
    maskWidth: number,
    maskHeight: number,
    canvasWidth: number,
    canvasHeight: number
) {
  const offscreen = document.createElement("canvas");
  offscreen.width = canvasWidth;
  offscreen.height = canvasHeight;
  const offCtx = offscreen.getContext("2d")!;
  const scaleX = canvasWidth / maskWidth;
  const scaleY = canvasHeight / maskHeight;

  const wallTexture = await loadTexture("/Textures/Floor.webp");
  const floorTexture = await loadTexture("/Textures/Floor1.jpg");

  const wallTileScale = 0.25;
  const floorTileScale = 0.3;

  const wallPatternCanvas = createTiledPattern(wallTexture, wallTileScale);
  const floorPatternCanvas = createTiledPattern(floorTexture, floorTileScale);

  const wallPattern = offCtx.createPattern(wallPatternCanvas, "repeat")!;
  const floorPattern = offCtx.createPattern(floorPatternCanvas, "repeat")!;

  // --- WALL FIX START ---
  const wallMask = new Uint8Array(outputClasses.length);
  for (let i = 0; i < outputClasses.length; i++) {
    if (outputClasses[i] === 0) {
      wallMask[i] = 1;
    }
  }

  const cleanWallMask = dilate(wallMask, maskWidth, maskHeight);

  offCtx.save();
  offCtx.setTransform(1, 0, 0, 1, 0, 0);
  offCtx.fillStyle = wallPattern;
  for (let y = 0; y < maskHeight; y++) {
    for (let x = 0; x < maskWidth; x++) {
      const idx = y * maskWidth + x;
      if (cleanWallMask[idx]) {
        offCtx.globalAlpha = 0.85;
        offCtx.fillRect(x * scaleX, y * scaleY, scaleX, scaleY);
      }
    }
  }
  offCtx.restore();
  // --- WALL FIX END ---

  // --- FLOOR FIX START ---
  const floorMask = new Uint8Array(outputClasses.length);
  for (let i = 0; i < outputClasses.length; i++) {
    if (outputClasses[i] === 3) {
      floorMask[i] = 1;
    }
  }

  const cleanFloorMask = dilate(floorMask, maskWidth, maskHeight);


  offCtx.save();
  offCtx.setTransform(1, 0.4, 0, 1, 0, -maskHeight * 0.2);
  offCtx.fillStyle = floorPattern;
  offCtx.globalAlpha = 0.85;
  for (let y = 0; y < maskHeight; y++) {
    for (let x = 0; x < maskWidth; x++) {
      const idx = y * maskWidth + x;
      if (cleanFloorMask[idx]) {
        offCtx.fillRect(x * scaleX, y * scaleY, scaleX, scaleY);
      }
    }
  }
  offCtx.restore();
  // --- FLOOR FIX END ---

  ctx.drawImage(offscreen, 0, 0, canvasWidth, canvasHeight);
}

async function run(image: HTMLImageElement) {
  const session = await loadModel();
  if (!session) {
    console.error("ONNX model not loaded.");
    return;
  }

  const width = 512;
  const height = 512;
  canvas.width = width;
  canvas.height = height;
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(image, 0, 0, width, height);

  const inputTensorData = preprocessImage(image);
  const tensor = new ort.Tensor("float32", inputTensorData, [1, 3, width, height]);

  const feeds = { input: tensor };
  const results = await session.run(feeds);

  const output = results[Object.keys(results)[0]];
  const numClasses = output.dims[1];
  const outHeight = output.dims[2];
  const outWidth = output.dims[3];
  const outputData = output.data as Float32Array;

  const classIds = new Uint8Array(outWidth * outHeight);
  for (let pixelIndex = 0; pixelIndex < outWidth * outHeight; pixelIndex++) {
    let maxScore = -Infinity;
    let maxClassId = 0;

    for (let classIndex = 0; classIndex < numClasses; classIndex++) {
      const score = outputData[classIndex * outWidth * outHeight + pixelIndex];
      if (score > maxScore) {
        maxScore = score;
        maxClassId = classIndex;
      }
    }
    classIds[pixelIndex] = maxClassId;
  }

  await drawSegmentationOverlay(classIds, outWidth, outHeight, width, height);
}

imageInput.addEventListener("change", (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;

  if (!file.type.startsWith("image/")) {
    console.error("Unsupported file type:", file.type);
    return;
  }

  const objectURL = URL.createObjectURL(file);
  const img = new Image();
  img.src = objectURL;
  img.onload = () => {
    run(img);
    URL.revokeObjectURL(objectURL);
  };
});