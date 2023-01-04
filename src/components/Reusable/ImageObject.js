import React, {useState, useRef, useEffect} from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../../atoms/atoms';
import {Image, Card, Text} from '@chakra-ui/react'
import ContextMenu from './ContextMenu/ContextMenu';
import ContextMenuItem from './ContextMenu/ContextMenuItem';
import ContextMenuList from './ContextMenu/ContextMenuList';
import ContextMenuTrigger from './ContextMenu/ContextMenuTrigger';

export default function ImageObject ({b64, metadata}) {
    const [init_image, setInitImage] = useRecoilState(atom.initImageState);
    const [initImagePath, setInitImagePath] = useRecoilState(atom.initImagePathState);
    const [metadataJSON, setMetadataJSON] = useState({});

    const textRef = useRef();
    useEffect(()=>{
        if(metadata){
            setMetadataJSON(JSON.parse(metadata));
        }
    },[])


    const copyToClipboard = () => {
        window.copyToClipboard(b64);
    };
  
    return (
        <Card           
            style={{ cursor: 'pointer', backgroundColor: 'transparent' }}
            onMouseEnter={() => {
                textRef.current.style.visibility = "visible";
                textRef.current.previousSibling.style.filter = "brightness(30%)";
            }}
            onMouseLeave={() => {
                textRef.current.style.visibility = "hidden";
                textRef.current.previousSibling.style.filter = "brightness(100%)";
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
                    WebkitBoxOrient: 'vertical'
                  }}
                >
                {metadataJSON?.text_prompts}
            </Text>
        </Card>
      );
}
