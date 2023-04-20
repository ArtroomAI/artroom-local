export const getImageDimensions = async (base64: string) => {
    const img = new Image();
    img.src = base64;
    await img.decode();
    return { width: img.naturalWidth, height: img.naturalHeight };
};
