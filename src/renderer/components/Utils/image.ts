export const getImageDimensions = async (src: string) => {
    const img = await loadImage(src);
    return { width: img.naturalWidth, height: img.naturalHeight };
};

export const loadImage = async (src: string) => {
    const img = new Image();
    img.src = src;
    await img.decode();
    return img;
};
