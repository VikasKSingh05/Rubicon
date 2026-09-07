import mongoose from "mongoose";

export async function connectDb(uri) {
  await mongoose.connect(uri, { serverSelectionTimeoutMS: 5000 });
}

export async function disconnectDb() {
  await mongoose.disconnect();
}