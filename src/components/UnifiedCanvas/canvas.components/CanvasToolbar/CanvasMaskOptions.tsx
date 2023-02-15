import { FC } from 'react';
import { Flex, Button, Checkbox } from '@chakra-ui/react';
import { FaMask, FaTrash } from 'react-icons/fa';
import { useHotkeys } from 'react-hotkeys-hook';
import { useRecoilState, useRecoilValue, useSetRecoilState } from 'recoil';
import { Popover, IconButton } from '../../components';
import {
	layerAtom,
	maskColorAtom,
	shouldPreserveMaskedAreaAtom,
	clearMaskAction,
	setIsMaskEnabledAction,
	isStagingSelector,
} from '../../atoms/canvas.atoms';
import { RgbaColorPicker } from 'react-colorful';

export const CanvasMaskOptions: FC = () => {
	const [layer, setLayer] = useRecoilState(layerAtom);
	const [maskColor, setMaskColor] = useRecoilState(maskColorAtom);
	const [shouldPreserveMaskedArea, setShouldPreserveMaskedArea] =
		useRecoilState(shouldPreserveMaskedAreaAtom);
	const clearMask = useSetRecoilState(clearMaskAction);
	const [isMaskEnabled, setIsMaskEnabled] = useRecoilState(
		setIsMaskEnabledAction,
	);
	const isStaging = useRecoilValue(isStagingSelector);

	useHotkeys(
		['q'],
		() => {
			handleToggleMaskLayer();
		},
		{
			enabled: () => !isStaging,
			preventDefault: true,
		},
		[layer],
	);

	useHotkeys(
		['shift+c'],
		() => {
			handleClearMask();
		},
		{
			enabled: () => !isStaging,
			preventDefault: true,
		},
		[],
	);

	useHotkeys(
		['h'],
		() => {
			handleToggleEnableMask();
		},
		{
			enabled: () => !isStaging,
			preventDefault: true,
		},
		[isMaskEnabled],
	);

	const handleToggleMaskLayer = () => {
		setLayer(layer === 'mask' ? 'base' : 'mask');
	};

	const handleClearMask = () => clearMask();

	const handleToggleEnableMask = () => setIsMaskEnabled(!isMaskEnabled);

	return (
		<Popover
			trigger="hover"
			triggerComponent={
				<IconButton
					aria-label="Masking Options"
					tooltip="Masking Options"
					icon={<FaMask />}
					bg={layer === 'mask' ? '#ebb905' : '#b8babc'}
					isDisabled={isStaging}
				/>
			}>
			<Flex direction="column" gap="0.5rem">
				<Checkbox
					isChecked={isMaskEnabled}
					onChange={handleToggleEnableMask}>
					Enable Mask (H)
				</Checkbox>
				<Checkbox
					isChecked={shouldPreserveMaskedArea}
					onChange={e =>
						setShouldPreserveMaskedArea(e.target.checked)
					}>
					Preserve Masked Area
				</Checkbox>
				<RgbaColorPicker
					style={{
						paddingTop: '0.5rem',
						paddingBottom: '0.5rem',
						width: '100%',
					}}
					color={maskColor}
					onChange={newColor => setMaskColor(newColor)}
				/>
				<Button
					size="sm"
					leftIcon={<FaTrash />}
					onClick={handleClearMask}>
					Clear Mask (Shift+C)
				</Button>
			</Flex>
		</Popover>
	);
};
