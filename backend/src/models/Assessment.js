import mongoose from "mongoose";

export const ASSESSMENT_STATES = [
  "uploaded",
  "analyzing",
  "analyzed",
  "chain_pending",
  "chain_logged",
];

export const SEVERITY_BY_PREDICTION = {
  "None": "none",
  "Moderate": "moderate",
  "Severe Collapse": "severe",
};

const assessmentSchema = new mongoose.Schema(
  {
    user: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
    state: { type: String, enum: ASSESSMENT_STATES, default: "uploaded" },
    filename: {
      hsi: { type: String, default: null },
      lidar: { type: String, default: null },
    },
    prediction: { type: String, default: null },
    confidence: { type: Number, default: null },
    class_probs: { type: Map, of: Number, default: null },
    geojson_polygon: { type: Object, default: null },
    model_version: { type: String, default: null },
    hsiCid: { type: String, default: null },
    lidarCid: { type: String, default: null },
    txHash: { type: String, default: null },
    chainVerified: { type: Boolean, default: false },
    timestamps: {
      uploaded: { type: Date, default: null },
      analyzing: { type: Date, default: null },
      analyzed: { type: Date, default: null },
      chainPending: { type: Date, default: null },
      chainLogged: { type: Date, default: null },
    },
  },
  { timestamps: true },
);

assessmentSchema.virtual("severity").get(function severity() {
  return SEVERITY_BY_PREDICTION[this.prediction] ?? "none";
});

assessmentSchema.set("toJSON", {
  virtuals: true,
  versionKey: false,
  transform(_doc, ret) {
    ret.id = ret._id.toString();
    delete ret._id;
    return ret;
  },
});

export default mongoose.model("Assessment", assessmentSchema);