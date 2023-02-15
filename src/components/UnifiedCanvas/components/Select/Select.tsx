import {
	FormControl,
	FormLabel,
	Select as ChakraSelect,
	SelectProps,
	Tooltip,
	TooltipProps,
} from '@chakra-ui/react';
import { MouseEvent, FC } from 'react';

type ISelectProps = SelectProps & {
	label?: string;
	tooltip?: string;
	tooltipProps?: Omit<TooltipProps, 'children'>;
	validValues:
		| Array<number | string>
		| Array<{ key: string; value: string | number }>;
};
/**
 * Customized Chakra FormControl + Select multi-part component.
 */
export const Select: FC<ISelectProps> = props => {
	const {
		label,
		isDisabled,
		validValues,
		tooltip,
		tooltipProps,
		size = 'sm',
		fontSize = 'md',
		...rest
	} = props;
	return (
		<FormControl
			isDisabled={isDisabled}
			onClick={(e: MouseEvent<HTMLDivElement>) => {
				e.stopPropagation();
				e.nativeEvent.stopImmediatePropagation();
				e.nativeEvent.stopPropagation();
				e.nativeEvent.cancelBubble = true;
			}}>
			{label && (
				<FormLabel
					fontSize={fontSize}
					marginBottom={1}
					flexGrow={2}
					whiteSpace="nowrap">
					{label}
				</FormLabel>
			)}
			<Tooltip label={tooltip} {...tooltipProps}>
				<ChakraSelect
					borderRadius="5px"
					fontSize={fontSize}
					size={size}
					{...rest}>
					{validValues.map(opt => {
						return typeof opt === 'string' ||
							typeof opt === 'number' ? (
							<option key={opt} value={opt}>
								{opt}
							</option>
						) : (
							<option key={opt.value} value={opt.value}>
								{opt.key}
							</option>
						);
					})}
				</ChakraSelect>
			</Tooltip>
		</FormControl>
	);
};
