import React, {useState, useRef, useEffect} from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../../atoms/atoms';
import {Image, Card, Text} from '@chakra-ui/react'
import ContextMenu from './ContextMenu/ContextMenu';
import ContextMenuItem from './ContextMenu/ContextMenuItem';
import ContextMenuList from './ContextMenu/ContextMenuList';
import ContextMenuTrigger from './ContextMenu/ContextMenuTrigger';
import { ImageMetadata } from '../Modals/ImageModal/ImageModal';

export default function ImageObject ({b64, metadata} : { b64: string, metadata: string }) {
    const [init_image, setInitImage] = useRecoilState(atom.initImageState);
    const [initImagePath, setInitImagePath] = useRecoilState(atom.initImagePathState);
    const [metadataJSON, setMetadataJSON] = useState<ImageMetadata>({
        text_prompts: '',
        negative_prompts: '',
        W: '',
        H: '',
        seed: '',
        sampler: '',
        steps: '',
        strength: '',
        cfg_scale: '',
        ckpt: ''
    });

    const [showImageModal, setShowImageModal] = useRecoilState(atom.showImageModalState);
    const [imageModalB64, setImageModalB64] = useRecoilState(atom.imageModalB64State);
    const [imageModalMetadata, setImageModalMetadata] = useRecoilState(atom.imageModalMetadataState);

    const textRef = useRef<HTMLSpanElement & HTMLParagraphElement>();
    useEffect(()=>{
        if(metadata){
            setMetadataJSON(JSON.parse(metadata));
        }
    },[])


    const copyToClipboard = () => {
        window.api.copyToClipboard(b64);
    };
  
    return (
        <Card           
            onClick={()=>{
                setShowImageModal(true);
                setImageModalB64(b64);
                setImageModalMetadata(metadataJSON);
            }}
            style={{ cursor: 'pointer', backgroundColor: 'transparent' }}
            onMouseEnter={() => {
                if(textRef.current) {
                    textRef.current.style.visibility = "visible";
                    (textRef.current.previousSibling as HTMLImageElement).style.filter = "brightness(30%)";
                }
            }}
            onMouseLeave={() => {
                textRef.current.style.visibility = "hidden";
                (textRef.current.previousSibling as HTMLImageElement).style.filter = "brightness(100%)";
            }}>
          <Image src={b64} borderRadius="5%" width="600px" height="100%" objectFit="cover" />
          <Text
                ref={textRef}
                as="span"
                visibility="hidden"
                style={{
                    position: 'absolute',
                    bottom: 0,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    display: '-webkit-box',
                    WebkitLineClamp: '4',
                    WebkitBoxOrient: 'vertical',
                    padding: '15px 15px 0px 15px',
                    fontSize: '14px',
                    fontWeight: 'normal'
                  }}
                >
                {metadataJSON?.text_prompts}
            </Text>
        </Card>
      );
}
