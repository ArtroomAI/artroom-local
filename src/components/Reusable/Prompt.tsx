import React from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../../atoms/atoms';
import { DragDropFile } from './DragDropFile/DragDropFile';
import {
	Button,
	Center,
	Box,
	Flex,
	FormControl,
	FormLabel,
	HStack,
	Tooltip,
	VStack,
	Spacer,
} from '@chakra-ui/react';
import { FaQuestionCircle } from 'react-icons/fa';
import { AutoResizeTextarea } from './AutoResizeTextarea';

interface IPromptProps {
	setFocused: React.Dispatch<React.SetStateAction<boolean>>;
}

export const Prompt: React.FC<IPromptProps> = ({ setFocused }) => {
	const [imageSettings, setImageSettings] = useRecoilState(
		atom.imageSettingsState,
	);

	return (
		<>
			<Flex width="100%" align="baseline" justifyContent="space-between">
				<VStack w="80%">
					<FormControl>
						<HStack>
							<Tooltip
								fontSize="md"
								label="Type here what you want the AI to generate in the image"
								placement="top"
								shouldWrapChildren>
								<FaQuestionCircle color="#777" />
							</Tooltip>

							<FormLabel htmlFor="text_prompts">Prompt</FormLabel>

							<Spacer />

							{/* <Button h='25px' w = '100px' className='enhance-prompt' onClick={(event) => setPrompts(Use Prompt Enhance)}>
                Enhance
              </Button> */}
						</HStack>

						<Flex className="text-prompts">
							<AutoResizeTextarea
								id="text_prompts"
								name="text_prompts"
								onBlur={() => setFocused(false)}
								onChange={event =>
									setImageSettings({
										...imageSettings,
										text_prompts: event.target.value,
									})
								}
								onFocus={() => setFocused(true)}
								value={imageSettings.text_prompts}
								variant="outline"
							/>
						</Flex>
					</FormControl>

					<FormControl>
						<Flex>
							<HStack align="flex-end">
								<Tooltip
									fontSize="md"
									label="Type what you DON'T want the AI to generate in the image"
									placement="top"
									shouldWrapChildren>
									<Center h="24px">
										<FaQuestionCircle color="#777" />
									</Center>
								</Tooltip>

								<FormLabel htmlFor="negative_prompts">
									Negative Prompt
								</FormLabel>
							</HStack>

							<Spacer />

							<Box pb="10px">
								<Button
									className="default-negative-prompt"
									h="35px"
									onClick={() =>
										setImageSettings({
											...imageSettings,
											negative_prompts:
												'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry',
										})
									}
									w="150px">
									Default Negative
								</Button>
							</Box>
						</Flex>

						<Spacer />

						<Flex className="negative-prompts">
							<AutoResizeTextarea
								id="negative_prompts"
								name="negative_prompts"
								onBlur={() => setFocused(false)}
								onChange={event =>
									setImageSettings({
										...imageSettings,
										negative_prompts: event.target.value,
									})
								}
								onFocus={() => setFocused(true)}
								value={imageSettings.negative_prompts}
								variant="outline"
							/>
						</Flex>
					</FormControl>
				</VStack>

				<Box className="starting-image">
					<HStack>
						<Tooltip
							fontSize="md"
							label="Upload an image to use as the starting point instead of just random noise"
							placement="top"
							shouldWrapChildren>
							<FaQuestionCircle color="#777" />
						</Tooltip>

						<FormLabel htmlFor="init_image">
							Starting Image
						</FormLabel>
					</HStack>

					<Box paddingBottom={30}>
						<DragDropFile />
					</Box>
				</Box>
			</Flex>
		</>
	);
};
