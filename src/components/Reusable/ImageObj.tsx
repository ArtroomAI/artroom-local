import React from 'react';
import { useRecoilState, useRecoilValue, useSetRecoilState } from 'recoil';
import * as atom from '../../atoms/atoms';
import { Image } from '@chakra-ui/react';
import Logo from '../../assets/images/ArtroomLogoTransparent.png';
import LoadingGif from '../../assets/images/loading.gif';

import {
	ContextMenu,
	ContextMenuItem,
	ContextMenuList,
	ContextMenuTrigger,
} from './ContextMenu';

interface IImageObjProps {
	b64: string;
	active: boolean;
}

export const ImageObj: React.FC<IImageObjProps> = ({ b64, active }) => {
	const [imageSettings, setImageSettings] = useRecoilState(
		atom.imageSettingsState,
	);

	const queueRunning = useRecoilValue(atom.queueRunningState);
	const setInitImagePath = useSetRecoilState(atom.initImagePathState);

	const copyToClipboard = () => {
		alert('copy to clipboard placeholder');
		// window.api.copyToClipboard(b64);
	};

	return (
		<ContextMenu>
			<ContextMenuTrigger>
				{active ? (
					<Image
						alignSelf="center"
						fallbackSrc={queueRunning ? LoadingGif : Logo}
						fit="scale-down"
						h="55vh"
						src={b64}
					/>
				) : (
					<Image
						alignSelf="center"
						fit="scale-down"
						h="55vh"
						src={b64}
					/>
				)}
			</ContextMenuTrigger>

			<ContextMenuList>
				<ContextMenuItem
					onClick={() => {
						setInitImagePath('');
						setImageSettings({ ...imageSettings, init_image: b64 });
					}}
					disabled={false}>
					Set As Starting Image
				</ContextMenuItem>

				<ContextMenuItem
					onClick={() => {
						copyToClipboard();
					}}
					disabled={false}>
					Copy To Clipboard
				</ContextMenuItem>
			</ContextMenuList>
		</ContextMenu>
	);
};
