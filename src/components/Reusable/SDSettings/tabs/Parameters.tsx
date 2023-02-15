import React, { useEffect, useRef } from 'react';
import { useRecoilState, useRecoilValue } from 'recoil';
import * as atom from '../../../../atoms/atoms';
import {
	Flex,
	Box,
	Button,
	FormControl,
	FormLabel,
	Input,
	VStack,
	HStack,
	NumberInput,
	NumberInputField,
	Slider,
	SliderTrack,
	SliderFilledTrack,
	SliderThumb,
	Tooltip,
	Checkbox,
	Select,
	Spacer,
	Text,
	createStandaloneToast,
	useColorModeValue,
} from '@chakra-ui/react';
import { FaQuestionCircle } from 'react-icons/fa';

const aspect_ratios = [
	'None',
	'Init Image',
	'1:1',
	'1:2',
	'2:1',
	'4:3',
	'3:4',
	'16:9',
	'9:16',
	'Custom',
];

const samplers = [
	{ value: 'ddim', name: 'DDIM' },
	{ value: 'dpmpp_2m', name: 'DPM++ 2M Karras' },
	{ value: 'dpmpp_2s_ancestral', name: 'DPM++ 2S Ancestral Karras' },
	{ value: 'euler', name: 'Euler' },
	{ value: 'euler_a', name: 'Euler Ancestral' },
	{ value: 'dpm_2', name: 'DPM 2' },
	{ value: 'dpm_a', name: 'DPM 2 Ancestral' },
	{ value: 'lms', name: 'LMS' },
	{ value: 'heun', name: 'Heun' },
	{ value: 'plms', name: 'PLMS' },
];

export const Parameters: React.FC = () => {
	const [imageSettings, setImageSettings] = useRecoilState(
		atom.imageSettingsState,
	);
	const [aspectRatioSelection, setAspectRatioSelection] = useRecoilState(
		atom.aspectRatioSelectionState,
	);
	const ckpts = useRecoilValue(atom.ckptsState);
	const vaes = useRecoilValue(atom.vaesState);

	const { toast } = createStandaloneToast();

	const inputRef = useRef<HTMLInputElement>(null);

	useEffect(() => {
		if (imageSettings.width > 0) {
			let newHeight = imageSettings.height;
			if (
				aspectRatioSelection !== 'Init Image' &&
				aspectRatioSelection !== 'None'
			) {
				try {
					const values = imageSettings.aspect_ratio.split(':');
					const widthRatio = parseFloat(values[0]);
					const heightRatio = parseFloat(values[1]);
					if (!isNaN(widthRatio) && !isNaN(heightRatio)) {
						newHeight = Math.min(
							1920,
							Math.floor(
								(imageSettings.width * heightRatio) /
									widthRatio /
									64,
							) * 64,
						);
					}
				} catch (err) {
					console.log(err);
				}
				setImageSettings({ ...imageSettings, height: newHeight });
			}
		}
	}, [imageSettings.width, imageSettings.aspect_ratio]);

	const uploadSettings: React.ChangeEventHandler<HTMLInputElement> = e => {
		e.preventDefault();
		if (e.target.files && e.target.files[0]) {
			const fileReader = new FileReader();
			fileReader.readAsText(e.target.files[0], 'UTF-8');
			fileReader.onload = e => {
				if (e.target && !(e.target.result === '')) {
					try {
						const data = JSON.parse(e.target.result as string);
						let settings = data;
						if ('Settings' in data) {
							settings = data.Settings;
						}
						if (!('text_prompts' in settings)) {
							throw 'Invalid JSON';
						}
						setImageSettings({
							text_prompts: settings.text_prompts,
							negative_prompts: settings.negative_prompts,
							batch_name: settings.batch_name,
							n_iter: settings.n_iter,
							steps: settings.steps,
							strength: settings.strength,
							cfg_scale: settings.cfg_scale,
							sampler: settings.sampler,
							width: settings.width,
							height: settings.height,
							aspect_ratio: settings.aspect_ratio,
							ckpt: settings.ckpt,
							seed: settings.seed,
							speed: settings.speed,
							use_random_seed: settings.use_random_seed,
							init_image: settings.init_image,
							mask_image: '',
							invert: false,
							image_save_path: settings.image_save_path,
						});
					} catch (err) {
						toast({
							title: 'Load Failed',
							status: 'error',
							position: 'top',
							duration: 3000,
							isClosable: false,
							containerStyle: {
								pointerEvents: 'none',
							},
						});
					}
				}
			};
		}
	};

	return (
		<Flex>
			<Box rounded="md">
				<VStack className="sd-settings" spacing={3}>
					<FormControl className="folder-name-input">
						<FormLabel htmlFor="batch_name">
							Output Folder
						</FormLabel>

						<Input
							id="batch_name"
							name="batch_name"
							onChange={event =>
								setImageSettings({
									...imageSettings,
									batch_name: event.target.value,
								})
							}
							value={imageSettings.batch_name}
							variant="outline"
						/>
					</FormControl>

					<HStack>
						<FormControl className="num-images-input">
							<FormLabel htmlFor="n_iter">№ of Images</FormLabel>

							<NumberInput
								id="n_iter"
								min={1}
								name="n_iter"
								onChange={v => {
									setImageSettings({
										...imageSettings,
										n_iter: parseInt(v),
									});
								}}
								value={imageSettings.n_iter}
								variant="outline">
								<NumberInputField id="n_iter" />
							</NumberInput>
						</FormControl>

						<FormControl className="steps-input">
							<HStack>
								<FormLabel htmlFor="steps">
									№ of Steps
								</FormLabel>

								<Tooltip
									fontSize="md"
									label="Steps determine how long you want the model to spend on generating your image. The more steps you have, the longer it will take but you'll get better results. The results are less impactful the more steps you have, so you may stop seeing improvement after 100 steps. 50 is typically a good number"
									placement="left"
									shouldWrapChildren>
									<FaQuestionCircle color="#777" />
								</Tooltip>
							</HStack>

							<NumberInput
								id="steps"
								min={1}
								name="steps"
								onChange={v => {
									setImageSettings({
										...imageSettings,
										steps: parseInt(v),
									});
								}}
								value={imageSettings.steps}
								variant="outline">
								<NumberInputField id="steps" />
							</NumberInput>
						</FormControl>
					</HStack>

					<Box className="size-input" width="100%">
						<FormControl
							className=" aspect-ratio-input"
							marginBottom={2}>
							<FormLabel htmlFor="AspectRatio">
								Fixed Aspect Ratio
							</FormLabel>

							<HStack>
								<Select
									id="aspect_ratio_selection"
									name="aspect_ratio_selection"
									onChange={event => {
										setAspectRatioSelection(
											event.target.value,
										);
										if (event.target.value !== 'Custom') {
											setImageSettings({
												...imageSettings,
												aspect_ratio:
													event.target.value,
											});
										}
									}}
									bg={useColorModeValue(
										'initial',
										'background',
									)}
									colorScheme="cyan"
									value={aspectRatioSelection}
									variant="outline">
									{aspect_ratios.map(elem => (
										<option value={elem} key={elem}>
											{elem}
										</option>
									))}
								</Select>

								{aspectRatioSelection === 'Custom' ? (
									<Input
										id="aspect_ratio"
										name="aspect_ratio"
										onChange={event =>
											setImageSettings({
												...imageSettings,
												aspect_ratio:
													event.target.value,
											})
										}
										value={imageSettings.aspect_ratio}
										variant="outline"
									/>
								) : (
									<></>
								)}
							</HStack>
						</FormControl>

						<FormControl className="width-input">
							<FormLabel htmlFor="Width">Width:</FormLabel>

							<Slider
								colorScheme="teal"
								defaultValue={512}
								id="width"
								isReadOnly={
									imageSettings.aspect_ratio === 'Init Image'
								}
								max={1920}
								min={256}
								name="width"
								onChange={v => {
									setImageSettings({
										...imageSettings,
										width: parseInt(v.toString()),
									});
								}}
								step={64}
								value={imageSettings.width}
								variant="outline">
								<SliderTrack bg="#EEEEEE">
									<Box position="relative" right={10} />

									<SliderFilledTrack bg="buttonPrimary" />
								</SliderTrack>

								<Tooltip
									bg="buttonPrimary"
									color="white"
									isOpen={
										!(
											imageSettings.aspect_ratio ===
											'Init Image'
										)
									}
									label={`${imageSettings.width}`}
									placement="right">
									<SliderThumb />
								</Tooltip>
							</Slider>
						</FormControl>

						<FormControl className="height-input">
							<FormLabel htmlFor="Height">Height:</FormLabel>

							<Slider
								defaultValue={512}
								isReadOnly={
									imageSettings.aspect_ratio === 'Init Image'
								}
								max={1920}
								min={256}
								onChange={v => {
									setImageSettings({
										...imageSettings,
										height: parseInt(v.toString()),
									});
								}}
								step={64}
								value={imageSettings.height}>
								<SliderTrack bg="#EEEEEE">
									<Box position="relative" right={10} />

									<SliderFilledTrack bg="buttonPrimary" />
								</SliderTrack>

								<Tooltip
									bg="buttonPrimary"
									color="white"
									isOpen={
										!(
											imageSettings.aspect_ratio ===
											'Init Image'
										)
									}
									label={`${imageSettings.height}`}
									placement="right">
									<SliderThumb />
								</Tooltip>
							</Slider>
						</FormControl>
					</Box>

					<FormControl className="cfg-scale-input">
						<HStack>
							<FormLabel htmlFor="cfg_scale">
								Prompt Strength (CFG):
							</FormLabel>

							<Spacer />

							<Tooltip
								fontSize="md"
								label="Prompt Strength or CFG Scale determines how intense the generations are. A typical value is around 5-15 with higher numbers telling the AI to stay closer to the prompt you typed"
								placement="left"
								shouldWrapChildren>
								<FaQuestionCircle color="#777" />
							</Tooltip>
						</HStack>

						<NumberInput
							id="cfg_scale"
							min={0}
							name="cfg_scale"
							onChange={v => {
								setImageSettings({
									...imageSettings,
									cfg_scale: parseInt(v),
								});
							}}
							value={imageSettings.cfg_scale}
							variant="outline">
							<NumberInputField id="cfg_scale" />
						</NumberInput>
					</FormControl>

					{imageSettings.init_image.length > 0 ? (
						<FormControl className="strength-input">
							<HStack>
								<FormLabel htmlFor="Strength">
									Image Variation Strength:
								</FormLabel>

								<Spacer />

								<Tooltip
									fontSize="md"
									label="Strength determines how much your output will resemble your input image. Closer to 0 means it will look more like the original and closer to 1 means use more noise and make it look less like the input"
									placement="left"
									shouldWrapChildren>
									<FaQuestionCircle color="#777" />
								</Tooltip>
							</HStack>

							<Slider
								defaultValue={0.75}
								id="strength"
								isDisabled={
									imageSettings.init_image.length === 0
								}
								max={0.99}
								min={0.0}
								name="strength"
								onChange={v => {
									setImageSettings({
										...imageSettings,
										strength: parseFloat(v.toString()),
									});
								}}
								step={0.01}
								value={imageSettings.strength}
								variant="outline">
								<SliderTrack bg="#EEEEEE">
									<Box position="relative" right={10} />

									<SliderFilledTrack bg="buttonPrimary" />
								</SliderTrack>

								<Tooltip
									bg="buttonPrimary"
									color="white"
									isOpen={
										!(imageSettings.init_image.length === 0)
									}
									label={`${imageSettings.strength}`}
									placement="right">
									<SliderThumb />
								</Tooltip>
							</Slider>
						</FormControl>
					) : (
						<></>
					)}

					<FormControl className="samplers-input">
						<HStack>
							<FormLabel htmlFor="Sampler">Sampler</FormLabel>

							<Spacer />

							<Tooltip
								fontSize="md"
								label="Samplers determine how the AI model goes about the generation. Each sampler has its own aesthetic (sometimes they may even end up with the same results). Play around with them and see which ones you prefer!"
								placement="left"
								shouldWrapChildren>
								<FaQuestionCircle color="#777" />
							</Tooltip>
						</HStack>

						<Select
							id="sampler"
							name="sampler"
							onChange={event =>
								setImageSettings({
									...imageSettings,
									sampler: event.target.value,
								})
							}
							bg={useColorModeValue('initial', 'background')}
							value={imageSettings.sampler}
							variant="outline">
							{samplers.map(elem => (
								<option value={elem.value} key={elem.value}>
									{elem.name}
								</option>
							))}
						</Select>
					</FormControl>

					<FormControl className="model-ckpt-input">
						<FormLabel htmlFor="Ckpt">
							<HStack>
								<Text>Model</Text>
							</HStack>
						</FormLabel>
						<Select
							id="ckpt"
							name="ckpt"
							onChange={event =>
								setImageSettings({
									...imageSettings,
									ckpt: event.target.value,
								})
							}
							bg={useColorModeValue('initial', 'background')}
							value={imageSettings.ckpt}
							variant="outline">
							{ckpts?.length > 0 ? (
								<option value="">
									Choose Your Model Weights
								</option>
							) : (
								<></>
							)}

							{ckpts?.map((ckpt_option, i) => (
								<option key={i} value={ckpt_option}>
									{ckpt_option}
								</option>
							))}
						</Select>
					</FormControl>

					<FormControl className="model-ckpt-input">
						<FormLabel htmlFor="Ckpt">
							<HStack>
								<Text>Vae</Text>
							</HStack>
						</FormLabel>
						<Select
							id="vae"
							name="vae"
							onChange={event =>
								setImageSettings({
									...imageSettings,
									vae: event.target.value,
								})
							}
							bg={useColorModeValue('initial', 'background')}
							value={imageSettings.vae}
							variant="outline">
							<option value="">Choose Your Vae</option>

							{vaes?.map((vae_option, i) => (
								<option key={i} value={vae_option}>
									{vae_option}
								</option>
							))}
						</Select>
					</FormControl>

					<HStack className="seed-input">
						<FormControl>
							<HStack>
								<FormLabel htmlFor="seed">Seed:</FormLabel>

								<Spacer />

								<Tooltip
									fontSize="md"
									label="Seed controls randomness. If you set the same seed each time and use the same settings, then you will get the same results"
									placement="left"
									shouldWrapChildren>
									<FaQuestionCircle color="#777" />
								</Tooltip>
							</HStack>

							<NumberInput
								id="seed"
								isDisabled={imageSettings.use_random_seed}
								min={0}
								name="seed"
								onChange={v => {
									setImageSettings({
										...imageSettings,
										seed: parseInt(v),
									});
								}}
								value={imageSettings.seed}
								variant="outline">
								<NumberInputField id="seed" />
							</NumberInput>
						</FormControl>

						<VStack align="center" justify="center">
							<FormLabel htmlFor="use_random_seed" pb="3px">
								Random
							</FormLabel>
							<Checkbox
								id="use_random_seed"
								isChecked={imageSettings.use_random_seed}
								name="use_random_seed"
								onChange={() => {
									setImageSettings({
										...imageSettings,
										use_random_seed:
											!imageSettings.use_random_seed,
									});
								}}
								pb="12px"
							/>
						</VStack>
					</HStack>
					<input
						// accept="image/png, image/jpeg"
						accept="application/JSON"
						className="input-file-upload"
						multiple={false}
						onChange={uploadSettings}
						ref={inputRef}
						type="file"
					/>
					<Button
						className="load-settings-button"
						// onClick={uploadSettings}
						onClick={() => inputRef.current?.click()}
						w="100%">
						Load Settings
					</Button>
				</VStack>
			</Box>
		</Flex>
	);
};
