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

  // Draw walls
  offCtx.save();
  offCtx.fillStyle = wallPattern;
  for (let y = 0; y < maskHeight; y++) {
    for (let x = 0; x < maskWidth; x++) {
      const idx = y * maskWidth + x;
      if (outputClasses[idx] === 0) {
        offCtx.fillRect(x * scaleX, y * scaleY, scaleX, scaleY);
      }
    }
  }
  offCtx.restore();

  // Draw floor with depth perspective
  offCtx.save();
  offCtx.setTransform(1, 0.4, 0, 1, 0, -maskHeight * 0.2);
  offCtx.fillStyle = floorPattern;
  for (let y = 0; y < maskHeight; y++) {
    for (let x = 0; x < maskWidth; x++) {
      const idx = y * maskWidth + x;
      if (outputClasses[idx] === 3) {
        offCtx.fillRect(x * scaleX, y * scaleY, scaleX, scaleY);
      }
    }
  }
  offCtx.restore();

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


// import * as ort from "onnxruntime-web";
// import { env } from "onnxruntime-web";
//
// const canvas = document.getElementById("canvas") as HTMLCanvasElement;
// const ctx = canvas.getContext("2d")!;
// const imageInput = document.getElementById("imageInput") as HTMLInputElement;
//
// env.wasm.wasmPaths = "/ort/";
//
// // Load floor texture
// const floorTexture = new Image();
// floorTexture.src = "/Textures/Floor1.jpg";
//
// async function loadModel() {
//   console.log("Loading ONNX model...", ort);
//
//   if (!ort.env.wasm) {
//     console.error("WASM execution provider is not available.");
//     return null;
//   }
//
//   try {
//     const filePath = "../ONNX/segformer.onnx";
//     const session = await ort.InferenceSession.create(filePath, {
//       executionProviders: ["wasm"],
//     });
//     console.log("Model loaded successfully", session);
//     return session;
//   } catch (error) {
//     console.error("Error loading model:", error);
//     throw error;
//   }
// }
//
// function preprocessImage(image: HTMLImageElement): Float32Array {
//   const width = 512;
//   const height = 512;
//   canvas.width = width;
//   canvas.height = height;
//
//   ctx.drawImage(image, 0, 0, width, height);
//   const { data } = ctx.getImageData(0, 0, width, height);
//
//   const floatData = new Float32Array(3 * width * height);
//   for (let i = 0; i < width * height; i++) {
//     floatData[i] = data[i * 4] / 255;
//     floatData[i + width * height] = data[i * 4 + 1] / 255;
//     floatData[i + 2 * width * height] = data[i * 4 + 2] / 255;
//   }
//
//   return floatData;
// }
//
// function drawSegmentationOverlay(
//   outputClasses: Uint8Array,
//   maskWidth: number,
//   maskHeight: number,
//   canvasWidth: number,
//   canvasHeight: number
// ) {
//   const offscreen = document.createElement("canvas");
//   offscreen.width = maskWidth;
//   offscreen.height = maskHeight;
//   const offCtx = offscreen.getContext("2d");
//   if (!offCtx) {
//     console.error("Offscreen canvas 2D context is null.");
//     return;
//   }
//
//   // Draw wall mask (red with transparency)
//   const wallMask = new Uint8ClampedArray(maskWidth * maskHeight * 4);
//   const floorMask = new Uint8ClampedArray(maskWidth * maskHeight * 4);
//
//   for (let i = 0; i < maskWidth * maskHeight; i++) {
//     const classId = outputClasses[i];
//     const offset = i * 4;
//
//     if (classId === 0) {
//       // Wall: red + alpha
//       wallMask[offset] = 255;
//       wallMask[offset + 1] = 0;
//       wallMask[offset + 2] = 0;
//       wallMask[offset + 3] = 150;
//     } else if (classId === 3) {
//       // Floor: alpha only (mask for texture)
//       floorMask[offset + 3] = 255;
//     }
//   }
//
//   // Step 1: Draw wall mask
//   const wallImage = new ImageData(wallMask, maskWidth, maskHeight);
//   offCtx.putImageData(wallImage, 0, 0);
//
//   // Step 2: Draw floor texture clipped with alpha mask
//   if (floorTexture.complete && floorTexture.naturalWidth > 0) {
//     const floorPattern = offCtx.createPattern(floorTexture, "repeat");
//
//     if (floorPattern) {
//       const floorMaskCanvas = document.createElement("canvas");
//       floorMaskCanvas.width = maskWidth;
//       floorMaskCanvas.height = maskHeight;
//       const floorMaskCtx = floorMaskCanvas.getContext("2d")!;
//       floorMaskCtx.putImageData(new ImageData(floorMask, maskWidth, maskHeight), 0, 0);
//
//       // Draw texture
//       offCtx.save();
//       offCtx.globalAlpha = 0.8; // adjust opacity as needed
//       offCtx.fillStyle = floorPattern;
//       offCtx.fillRect(0, 0, maskWidth, maskHeight);
//       offCtx.globalCompositeOperation = "destination-in"; // keep only where mask is
//       offCtx.drawImage(floorMaskCanvas, 0, 0);
//       offCtx.restore();
//     }
//   } else {
//     console.warn("Floor texture not yet loaded.");
//   }
//
//   // Final: Draw result on main canvas (scaled)
//   ctx.drawImage(offscreen, 0, 0, canvasWidth, canvasHeight);
// }
//
//
//
// function getColorForClass(classId: number): [number, number, number] {
//   if (classId === 0) return [255, 0, 0]; // wall - red
//   if (classId === 3) return [0, 0, 255]; // floor - was blue (now textured)
//   return [0, 0, 0]; // background
// }
//
// async function run(image: HTMLImageElement) {
//   console.log("Running model on image", image);
//   const session = await loadModel();
//   if (!session) {
//     console.error("Failed to load ONNX model session.");
//     return;
//   }
//
//   const width = 512;
//   const height = 512;
//   canvas.width = width;
//   canvas.height = height;
//   ctx.clearRect(0, 0, width, height);
//   ctx.drawImage(image, 0, 0, width, height);
//
//   const inputTensorData = preprocessImage(image);
//   const tensor = new ort.Tensor("float32", inputTensorData, [1, 3, width, height]);
//
//   const feeds = { input: tensor };
//   const results = await session.run(feeds);
//   console.log("Model inference results:", results);
//
//   const output = results[Object.keys(results)[0]];
//   const numClasses = output.dims[1];
//   const outHeight = output.dims[2];
//   const outWidth = output.dims[3];
//
//   const outputData = output.data as Float32Array;
//   const classIds = new Uint8Array(outWidth * outHeight);
//
//   for (let pixelIndex = 0; pixelIndex < outWidth * outHeight; pixelIndex++) {
//     let maxScore = -Infinity;
//     let maxClassId = 0;
//
//     for (let classIndex = 0; classIndex < numClasses; classIndex++) {
//       const score = outputData[classIndex * outWidth * outHeight + pixelIndex];
//       if (score > maxScore) {
//         maxScore = score;
//         maxClassId = classIndex;
//       }
//     }
//     classIds[pixelIndex] = maxClassId;
//   }
//
//   drawSegmentationOverlay(classIds, outWidth, outHeight, width, height);
// }
//
//
// imageInput.addEventListener("change", (e) => {
//   const file = (e.target as HTMLInputElement).files?.[0];
//   if (!file) return;
//
//   if (!file.type.startsWith("image/")) {
//     console.error("Unsupported file type:", file.type);
//     return;
//   }
//
//   const objectURL = URL.createObjectURL(file);
//   const img = new Image();
//   img.src = objectURL;
//   img.onload = () => {
//     run(img);
//     URL.revokeObjectURL(objectURL);
//   };
// });
