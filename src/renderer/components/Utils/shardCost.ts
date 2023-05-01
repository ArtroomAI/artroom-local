export const computeShardCost = (imageSettings: QueueType) => {
    const { width, height, steps, n_iter } = imageSettings;
    const estimated_price = (width * height) / (512 * 512) * (steps / 50) * n_iter * 10;
    return Math.round(estimated_price);
}
