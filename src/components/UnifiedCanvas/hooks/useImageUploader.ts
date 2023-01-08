let openFunction: () => void;

export const useImageUploader = () => {
  return {
    setOpenUploader: (open?: () => void) => {
      if (open) {
        openFunction = open;
      }
    },
    openUploader: openFunction,
  };
};
