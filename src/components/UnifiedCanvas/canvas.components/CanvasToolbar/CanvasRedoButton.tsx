import { FC } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { FaRedo } from 'react-icons/fa';
import { useRecoilValue, useSetRecoilState } from 'recoil';
import { IconButton } from '../../components';
import { redoAction, canRedoSelector } from '../../atoms/canvas.atoms';

export const CanvasRedoButton: FC = () => {
	const redo = useSetRecoilState(redoAction);
	const canRedo = useRecoilValue(canRedoSelector);

	const handleRedo = () => {
		redo();
	};

	useHotkeys(
		['meta+shift+z', 'ctrl+shift+z', 'control+y', 'meta+y'],
		() => {
			handleRedo();
		},
		{
			enabled: () => canRedo,
			preventDefault: true,
		},
		[canRedo],
	);

	return (
		<IconButton
			aria-label="Redo (Ctrl+Shift+Z)"
			tooltip="Redo (Ctrl+Shift+Z)"
			icon={<FaRedo />}
			onClick={handleRedo}
			isDisabled={!canRedo}
		/>
	);
};
