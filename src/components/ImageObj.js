import React from 'react'
import { useRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import {Image} from '@chakra-ui/react'
import Logo from '../images/ArtroomLogoTransparent.png';
import LoadingGif from '../images/loading.gif';

import ContextMenu from './ContextMenu/ContextMenu';
import ContextMenuItem from './ContextMenu/ContextMenuItem';
import ContextMenuList from './ContextMenu/ContextMenuList';
import ContextMenuTrigger from './ContextMenu/ContextMenuTrigger';

export default function ImageObj({B64, active}) {
    const [queueRunning, setQueueRunning] = useRecoilState(atom.queueRunningState);
    const [init_image, setInitImage] = useRecoilState(atom.initImageState);
    const [initImagePath, setInitImagePath] = useRecoilState(atom.initImagePathState);

    const copyToClipboard = () => {
        window['copyToClipboard'](B64);
      }

    return (
        <ContextMenu>
            <ContextMenuTrigger>
                {active ?
                    <Image h='55vh' fit='scale-down' src={B64} fallbackSrc= {queueRunning ? LoadingGif : Logo}/>
                    :
                    <Image h='55vh' fit='scale-down' src={B64}/>
                }
            </ContextMenuTrigger>
            <ContextMenuList>
                <ContextMenuItem onClick={() => {
                    setInitImagePath('');
                    setInitImage(B64);
                    }
                }>
                    Set As Starting Image
                </ContextMenuItem>
                <ContextMenuItem onClick={() => {
                    copyToClipboard();
                    }
                }>
                    Copy To Clipboard
                </ContextMenuItem>
            </ContextMenuList>
        </ContextMenu>
    );
}