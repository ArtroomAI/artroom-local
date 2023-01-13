import React, {
  useCallback,
  ReactNode,
  useState,
  useEffect,
  KeyboardEvent,
  FC,
} from 'react';
import { FileRejection, useDropzone } from 'react-dropzone';
import { useToast } from '@chakra-ui/react';
import { useImageUploader } from '../../hooks';
import { ImageUploaderTriggerContext } from './ImageUploaderTriggerContext';
import { ImageUploadOverlay } from './ImageUploadOverlay';
import { uploadImage } from '../../helpers/uploadImage';
import { useSetRecoilState, useRecoilState, useRecoilValue } from 'recoil';
import { boundingBoxCoordinatesAtom, boundingBoxDimensionsAtom, futureLayerStatesAtom, layerStateAtom, maxHistoryAtom, pastLayerStatesAtom, setInitialCanvasImageAction } from '../../atoms/canvas.atoms';
import _ from 'lodash';

type ImageUploaderProps = {
  children: ReactNode;
};

export const ImageUploader: FC<ImageUploaderProps> = (props) => {
  const { children } = props;
  const toast = useToast({});
  const [isHandlingUpload, setIsHandlingUpload] = useState<boolean>(false);

  const boundingBoxCoordinates = useRecoilValue(boundingBoxCoordinatesAtom);  
  const boundingBoxDimensions = useRecoilValue(boundingBoxDimensionsAtom);  
  const maxHistory = useRecoilValue(maxHistoryAtom);  
  const [layerState, setLayerState] = useRecoilState(layerStateAtom);  
  const [pastLayerStates, setPastLayerStates] = useRecoilState(pastLayerStatesAtom);  
  const setFutureLayerStates = useSetRecoilState(futureLayerStatesAtom);  
  const setInitialCanvasImage = useSetRecoilState(setInitialCanvasImageAction)


  const { setOpenUploader } = useImageUploader();

  const fileRejectionCallback = useCallback(
    (rejection: FileRejection) => {
      setIsHandlingUpload(true);
      const msg = rejection.errors.reduce(
        (acc: string, cur: { message: string }) => `${acc}\n${cur.message}`,
        ''
      );
      toast({
        title: 'Upload failed',
        description: msg,
        status: 'error',
        isClosable: true,
      });
    },
    [toast]
  );

  const fileAcceptedCallback = useCallback(async (file: File) => {

    uploadImage({
      imageFile: file,
      setInitialCanvasImage,
      boundingBoxCoordinates,
      boundingBoxDimensions,
      setPastLayerStates,
      pastLayerStates,
      layerState,
      maxHistory,
      setLayerState,
      setFutureLayerStates
    });
  }, []);

  const onDrop = useCallback(
    (acceptedFiles: Array<File>, fileRejections: Array<FileRejection>) => {
      fileRejections.forEach((rejection: FileRejection) => {
        fileRejectionCallback(rejection);
      });
      acceptedFiles.forEach((file: File) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = () => {
          const b64 = reader.result;
          file["b64"] = b64;
          fileAcceptedCallback(file);
        }        
      });
    },
    [fileAcceptedCallback, fileRejectionCallback]
  );

  const {
    getRootProps,
    getInputProps,
    isDragAccept,
    isDragReject,
    isDragActive,
    open,
  } = useDropzone({
    accept: {
      'image/png': ['.png'],
      'image/jpeg': ['.jpg', '.jpeg', '.png'],
    },
    noClick: true,
    onDrop,
    onDragOver: () => setIsHandlingUpload(true),
    maxFiles: 1,
  });

  setOpenUploader(open);

  useEffect(() => {
    const pasteImageListener = (e: ClipboardEvent) => {
      const dataTransferItemList = e.clipboardData?.items;
      if (!dataTransferItemList) return;

      const imageItems: Array<DataTransferItem> = [];

      for (const item of dataTransferItemList) {
        if (
          item.kind === 'file' &&
          ['image/png', 'image/jpg'].includes(item.type)
        ) {
          imageItems.push(item);
        }
      }

      if (!imageItems.length) return;

      e.stopImmediatePropagation();

      if (imageItems.length > 1) {
        toast({
          description:
            'Multiple images pasted, may only upload one image at a time',
          status: 'error',
          isClosable: true,
        });
        return;
      }
      const file = imageItems[0].getAsFile();

      if (!file) {
        toast({
          description: 'Unable to load file',
          status: 'error',
          isClosable: true,
        });
        return;
      }
      uploadImage({
        imageFile: file,
        setInitialCanvasImage,
        boundingBoxCoordinates,
        boundingBoxDimensions,
        setPastLayerStates,
        pastLayerStates,
        layerState,
        maxHistory,
        setLayerState,
        setFutureLayerStates
      });
    };
    document.addEventListener('paste', pasteImageListener);
    return () => {
      document.removeEventListener('paste', pasteImageListener);
    };
  }, [toast]);

  const overlaySecondaryText = ' to Unified Canvas';

  return (
    <ImageUploaderTriggerContext.Provider value={open}>
      <div
        {...getRootProps({ style: {} })}
        onKeyDown={(e: KeyboardEvent) => {
          // Bail out if user hits spacebar - do not open the uploader
          if (e.key === ' ') return;
        }}
      >
        <input {...getInputProps()} />
        {children}
        {isDragActive && isHandlingUpload && (
          <ImageUploadOverlay
            isDragAccept={isDragAccept}
            isDragReject={isDragReject}
            overlaySecondaryText={overlaySecondaryText}
            setIsHandlingUpload={setIsHandlingUpload}
          />
        )}
      </div>
    </ImageUploaderTriggerContext.Provider>
  );
};
