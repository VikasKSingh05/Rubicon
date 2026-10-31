import mongoose from "mongoose";
import User from "../src/models/User.js";
import Assessment from "../src/models/Assessment.js";
import { config } from "../src/config.js";

export const XBD_SCENES = [
  {
    disaster: "santa-rosa-wildfire",
    name: "Santa Rosa Wildfire (California)",
    tileId: "00000001",
    center: { lng: -122.7141, lat: 38.4404 },
    prediction: "Destroyed",
    confidence: 0.942,
    class_probs: { "No Damage": 0.051, "Minor Damage": 0.076, "Major Damage": 0.171, "Destroyed": 0.702 },
    buildingCount: 14,
    preCid: "QmQmc6NSg1Bvv8vSrkeS6DQnxa8m2YyP8H9V3Mp3p4F3TWHM",
    postCid: "QmQmYzh18FjRahBexayJTr74U6r72cxR1NvBBam1ZxzuNsge",
    txHash: "0x7a8c43912a7d65df19f56e9c402b8d003ac5b9e112d78e3c88bb7201c9053fb2",
  },
  {
    disaster: "hurricane-michael",
    name: "Hurricane Michael (Mexico Beach, FL)",
    tileId: "00000014",
    center: { lng: -85.4216, lat: 29.9438 },
    prediction: "Destroyed",
    confidence: 0.965,
    class_probs: { "No Damage": 0.021, "Minor Damage": 0.048, "Major Damage": 0.151, "Destroyed": 0.780 },
    buildingCount: 16,
    preCid: "QmR5z2K7tXhD5Lw9QvB9h2F7mE6wT8kC1pY4uV3bZ8oN2a",
    postCid: "QmT6b7K1rW9yL2oP4xZ8mB3vE5wQ9hY7uF1nV2sC4tM3d",
    txHash: "0x5e2b9c7104f18d6a89c3a078125439bbdeac98721495c3207865ae2187b5a19c",
  },
  {
    disaster: "mexico-earthquake",
    name: "Mexico City Earthquake (Puebla M7.1)",
    tileId: "00000021",
    center: { lng: -99.1500, lat: 19.3400 },
    prediction: "Major Damage",
    confidence: 0.895,
    class_probs: { "No Damage": 0.152, "Minor Damage": 0.218, "Major Damage": 0.510, "Destroyed": 0.120 },
    buildingCount: 12,
    preCid: "QmP7x3Y9uK2oB4wV6mE8hZ1rT5nV9sL3bF7mQ1tW8yC2d",
    postCid: "QmW2k8B5vT7yL9oP1mZ3hE6rQ4nV8sF2tM9bY1uX7cR4a",
    txHash: "0x892a0b1c784e93d56a7c8102374b5c689d0e123fa4567890bcde1234567890ab",
  },
  {
    disaster: "volcano-fuego",
    name: "Volcán de Fuego Pyroclastic Flow (Guatemala)",
    tileId: "00000030",
    center: { lng: -90.8808, lat: 14.4747 },
    prediction: "Destroyed",
    confidence: 0.938,
    class_probs: { "No Damage": 0.038, "Minor Damage": 0.062, "Major Damage": 0.198, "Destroyed": 0.702 },
    buildingCount: 10,
    preCid: "QmX9t2L5yK8oB1wV4mE6hZ7rT3nV1sL9bF5mQ2tW7yC4a",
    postCid: "QmY4m8B2vT9yL1oP6mZ5hE3rQ8nV2sF4tM1bY7uX9cR2d",
    txHash: "0x4c8f129a0b3e56d78a9c01234b5c689d0e123fa4567890bcde123456789012cd",
  },
  {
    disaster: "palu-tsunami",
    name: "Sulawesi Palu Earthquake & Tsunami (Indonesia)",
    tileId: "00000045",
    center: { lng: 119.8707, lat: -0.8917 },
    prediction: "Destroyed",
    confidence: 0.951,
    class_probs: { "No Damage": 0.031, "Minor Damage": 0.069, "Major Damage": 0.220, "Destroyed": 0.680 },
    buildingCount: 15,
    preCid: "QmZ1r9T4yK6oB8wV2mE4hZ5rT7nV3sL1bF9mQ4tW5yC6a",
    postCid: "QmV8s5F1tM3bY7uX9cR2dW2k8B5vT7yL9oP1mZ3hE6rQ",
    txHash: "0x3d7b891a2c4e56f78a9c01234b5c689d0e123fa4567890bcde123456789023ef",
  },
  {
    disaster: "midwest-floods",
    name: "Midwest River Basin Floods (Nebraska)",
    tileId: "00000050",
    center: { lng: -95.9345, lat: 41.2565 },
    prediction: "Minor Damage",
    confidence: 0.862,
    class_probs: { "No Damage": 0.320, "Minor Damage": 0.480, "Major Damage": 0.140, "Destroyed": 0.060 },
    buildingCount: 11,
    preCid: "QmB3vE5wQ9hY7uF1nV2sC4tM3dQmR5z2K7tXhD5Lw9Qv",
    postCid: "QmT5nV9sL3bF7mQ1tW8yC2dQmP7x3Y9uK2oB4wV6mE8h",
    txHash: "0x2e6a780b1c3d45e67f8a9b01234c5d689e0f123ab4567890cdef123456789012",
  },
  {
    disaster: "joplin-tornado",
    name: "Joplin EF5 Tornado Corridor (Missouri)",
    tileId: "00000080",
    center: { lng: -94.5133, lat: 37.0842 },
    prediction: "Destroyed",
    confidence: 0.971,
    class_probs: { "No Damage": 0.012, "Minor Damage": 0.038, "Major Damage": 0.178, "Destroyed": 0.772 },
    buildingCount: 14,
    preCid: "QmF7mQ2tW7yC4aQmX9t2L5yK8oB1wV4mE6hZ7rT3nV1s",
    postCid: "QmY7uX9cR2dQmY4m8B2vT9yL1oP6mZ5hE3rQ8nV2sF4t",
    txHash: "0x1f5c678a9b0d12e34f5a6b78901c2d345e6f789ab0123456cdef789012345678",
  },
  {
    disaster: "woolsey-fire",
    name: "Woolsey Wildfire (Malibu / Los Angeles, CA)",
    tileId: "00000060",
    center: { lng: -118.7798, lat: 34.0259 },
    prediction: "Major Damage",
    confidence: 0.912,
    class_probs: { "No Damage": 0.120, "Minor Damage": 0.180, "Major Damage": 0.520, "Destroyed": 0.180 },
    buildingCount: 13,
    preCid: "QmW8yC2dQmP7x3Y9uK2oB4wV6mE8hZ1rT5nV9sL3bF7m",
    postCid: "QmZ3hE6rQ4nV8sF2tM9bY1uX7cR4aQmW2k8B5vT7yL9o",
    txHash: "0x9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c2d1e0f9a8b",
  },
  {
    disaster: "pinery-bushfire",
    name: "Pinery Bushfire (South Australia)",
    tileId: "00000070",
    center: { lng: 138.5667, lat: -34.3317 },
    prediction: "Major Damage",
    confidence: 0.889,
    class_probs: { "No Damage": 0.180, "Minor Damage": 0.240, "Major Damage": 0.460, "Destroyed": 0.120 },
    buildingCount: 9,
    preCid: "QmK2oB4wV6mE8hZ1rT5nV9sL3bF7mQ1tW8yC2dQmP7x3",
    postCid: "QmV4mE6hZ7rT3nV1sL9bF5mQ2tW7yC4aQmX9t2L5yK8o",
    txHash: "0x8f7e6d5c4b3a2f1e0d9c8b7a6f5e4d3c2b1a0f9e8d7c6b5a4f3e2d1c0b9a8f7e",
  },
  {
    disaster: "portugal-wildfire",
    name: "Pedrógão Grande Wildfire (Portugal)",
    tileId: "00000090",
    center: { lng: -8.1444, lat: 39.9167 },
    prediction: "Major Damage",
    confidence: 0.903,
    class_probs: { "No Damage": 0.140, "Minor Damage": 0.210, "Major Damage": 0.490, "Destroyed": 0.160 },
    buildingCount: 10,
    preCid: "QmL9bF5mQ2tW7yC4aQmX9t2L5yK8oB1wV4mE6hZ7rT3n",
    postCid: "QmP6mZ5hE3rQ8nV2sF4tM1bY7uX9cR2dQmY4m8B2vT9y",
    txHash: "0x7e6d5c4b3a2f1e0d9c8b7a6f5e4d3c2b1a0f9e8d7c6b5a4f3e2d1c0b9a8f7e6d",
  },
];

function makeBBoxPolygon(lng, lat, sizeDeg = 0.009) {
  const half = sizeDeg / 2;
  return {
    type: "Polygon",
    coordinates: [
      [
        [Number((lng - half).toFixed(6)), Number((lat - half).toFixed(6))],
        [Number((lng + half).toFixed(6)), Number((lat - half).toFixed(6))],
        [Number((lng + half).toFixed(6)), Number((lat + half).toFixed(6))],
        [Number((lng - half).toFixed(6)), Number((lat + half).toFixed(6))],
        [Number((lng - half).toFixed(6)), Number((lat - half).toFixed(6))],
      ],
    ],
  };
}

function makeBuildingFootprints(lng, lat, count, pred) {
  const features = [];
  const radius = 0.0035;
  const labels = pred === "Destroyed" 
    ? ["Destroyed", "Destroyed", "Major Damage", "Minor Damage", "No Damage"]
    : ["Major Damage", "Minor Damage", "Minor Damage", "No Damage", "No Damage"];

  for (let i = 0; i < count; i++) {
    const angle = (2 * Math.PI * i) / count;
    const r = radius * (0.3 + 0.7 * ((i % 3) / 2));
    const bx = lng + r * Math.cos(angle);
    const by = lat + r * Math.sin(angle);
    const bSize = 0.0003 + (i % 3) * 0.00015;
    const damage = labels[i % labels.length];

    features.push({
      type: "Feature",
      id: `b_${i}`,
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [Number((bx - bSize).toFixed(6)), Number((by - bSize).toFixed(6))],
            [Number((bx + bSize).toFixed(6)), Number((by - bSize).toFixed(6))],
            [Number((bx + bSize).toFixed(6)), Number((by + bSize).toFixed(6))],
            [Number((bx - bSize).toFixed(6)), Number((by + bSize).toFixed(6))],
            [Number((bx - bSize).toFixed(6)), Number((by - bSize).toFixed(6))],
          ],
        ],
      },
      properties: {
        feature_type: "building",
        subtype: damage,
        uid: `bld_${i}_${damage.toLowerCase().replace(/\s+/g, "_")}`,
        area_pixels: 420 + i * 85,
        confidence: Number((0.85 + (i % 10) * 0.012).toFixed(3)),
      },
    });
  }
  return { type: "FeatureCollection", features };
}

export async function seedXbdAssessments(targetUserId = null) {
  await mongoose.connect(config.mongoUri);
  console.log("Connected to MongoDB for xBD scene seeding...");

  let users = [];
  if (targetUserId) {
    const u = await User.findById(targetUserId);
    if (u) users = [u];
  } else {
    users = await User.find({});
  }

  if (users.length === 0) {
    // Create default demo user if none exists
    const demo = await User.create({
      email: "demo@rubicon.dev",
      password: "Password123",
    });
    users = [demo];
    console.log("Created demo user: demo@rubicon.dev / Password123");
  }

  let totalInserted = 0;

  for (const user of users) {
    for (const sc of XBD_SCENES) {
      const preName = `${sc.disaster}_${sc.tileId}_pre_disaster.png`;
      const postName = `${sc.disaster}_${sc.tileId}_post_disaster.png`;

      // Check if already seeded
      const existing = await Assessment.findOne({
        user: user._id,
        "filename.pre": preName,
      });

      if (!existing) {
        const bbox = makeBBoxPolygon(sc.center.lng, sc.center.lat);
        const buildings = makeBuildingFootprints(sc.center.lng, sc.center.lat, sc.buildingCount, sc.prediction);

        await Assessment.create({
          user: user._id,
          state: "chain_logged",
          filename: {
            pre: preName,
            post: postName,
          },
          storage: {
            pre: `xbd/${preName}`,
            post: `xbd/${postName}`,
          },
          prediction: sc.prediction,
          confidence: sc.confidence,
          class_probs: sc.class_probs,
          geojson_polygon: bbox,
          buildings_geojson: buildings,
          model_version: "xbd-siamese-unet-v1.0-resnet34",
          preCid: sc.preCid,
          postCid: sc.postCid,
          txHash: sc.txHash,
          chainVerified: true,
          timestamps: {
            uploaded: new Date(Date.now() - 3600000 * 24 * 3),
            analyzing: new Date(Date.now() - 3600000 * 24 * 3 + 1000),
            analyzed: new Date(Date.now() - 3600000 * 24 * 3 + 3000),
            chainPending: new Date(Date.now() - 3600000 * 24 * 3 + 4000),
            chainLogged: new Date(Date.now() - 3600000 * 24 * 3 + 6000),
          },
        });
        totalInserted++;
      }
    }
  }

  console.log(`Successfully seeded ${totalInserted} xBD trained scenes across ${users.length} user(s).`);
  return totalInserted;
}

if (process.argv[1]?.endsWith("seed_xbd.js")) {
  seedXbdAssessments()
    .then(() => process.exit(0))
    .catch((err) => {
      console.error("Seeding failed:", err);
      process.exit(1);
    });
}
