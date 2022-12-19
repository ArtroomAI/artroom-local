import React, { useReducer, useEffect } from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../../atoms/atoms';
import JoyRide, { ACTIONS, EVENTS, STATUS } from 'react-joyride';
import { useNavigate, useLocation } from 'react-router-dom';
import { FaQuestionCircle } from 'react-icons/fa';
import {
    MenuButton,
    HStack,
    Icon,
    Text,
    Flex,
    Menu,
    Link,
    Tooltip
} from '@chakra-ui/react';
import { TOUR_STEPS_MAIN, TOUR_STEPS_PAINT, TOUR_STEPS_QUEUE, TOUR_STEPS_UPSCALE, TOUR_STEPS_SETTINGS } from './ToursSteps';
const INITIAL_STATE = {
    key: new Date(), // This field makes the tour to re-render when we restart the tour
    run: false,
    continuous: true,
    loading: false,
    stepIndex: 0,
    steps: TOUR_STEPS_MAIN
};

// Reducer will manage updating the local state
const reducer = (state = INITIAL_STATE, action) => {
    switch (action.type) {
    case '/':
        return { ...state,
            stepIndex: 0,
            run: true,
            steps: TOUR_STEPS_MAIN,
            loading: false,
            key: new Date() };
    case '/paint':
        return { ...state,
            stepIndex: 0,
            run: true,
            steps: TOUR_STEPS_PAINT,
            loading: false,
            key: new Date() };
    case '/queue':
        return { ...state,
            stepIndex: 0,
            run: true,
            steps: TOUR_STEPS_QUEUE,
            loading: false,
            key: new Date() };
    case '/upscale':
        return { ...state,
            stepIndex: 0,
            run: true,
            steps: TOUR_STEPS_UPSCALE,
            loading: false,
            key: new Date() };
    case '/settings':
        return { ...state,
            stepIndex: 0,
            run: true,
            steps: TOUR_STEPS_SETTINGS,
            loading: false,
            key: new Date() };
    case 'START':
        return { ...state,
            run: true,
            key: new Date() };
    case 'RESET':
        return { ...state,
            stepIndex: 0,
            key: new Date() };
    case 'STOP':
        return { ...state,
            run: false };
    case 'NEXT_OR_PREV':
        return { ...state,
            ...action.payload };
    case 'RESTART':
        return {
            ...state,
            stepIndex: 0,
            run: true,
            loading: false,
            key: new Date()
        };
    default:
        return state;
    }
};

// Tour component
const Tour = () => {
    // Tour state is the state which control the JoyRide component
    const [navSize, changeNavSize] = useRecoilState(atom.navSizeState);
    const [tourState, dispatch] = useReducer(
        reducer,
        INITIAL_STATE
    );
    const navigate = useNavigate();
    const location = useLocation();
    useEffect(
        () => {
            // Auto start the tour if the tour is not viewed before
            if (!localStorage.getItem('tour')) {
                dispatch({ type: 'START' });
            }
        },
        []
    );

    // Set once tour is viewed, skipped or closed
    const setTourViewed = () => {
        localStorage.setItem(
            'tour',
            '1'
        );
    };

    const callback = (data) => {
        const { action, index, type, status } = data;
        if (
        // If close button clicked, then close the tour
            action === ACTIONS.CLOSE ||
      // If skipped or end tour, then close the tour
      status === STATUS.SKIPPED && tourState.run ||
      status === STATUS.FINISHED
        ) {
            setTourViewed();
            dispatch({ type: 'STOP' });
        } else if (type === EVENTS.STEP_AFTER || type === EVENTS.TARGET_NOT_FOUND) {
            // Check whether next or back button click and update the step.
            dispatch({
                type: 'NEXT_OR_PREV',
                payload: { stepIndex: index + (action === ACTIONS.PREV
                    ? -1
                    : 1) }
            });
        }
    };

    const startTour = () => {
        dispatch({ type: location.pathname });
    };

    return (
        <>
            <Flex
                alignItems={navSize === 'small'
                    ? 'center'
                    : 'flex-start'}
                flexDir="column"
                mt={25}
                onClick={startTour}
                w="100%"
            >
                <Menu placement="right">
                    <Link
                        _hover={{ textDecor: 'none',
                            backgroundColor: '#AEC8CA' }}
                        borderRadius={8}
                        p={2.5}
                    >
                        <Tooltip
                            fontSize="md"
                            label={navSize === 'small'
                                ? 'Tutorial'
                                : ''}
                            placement="bottom"
                            shouldWrapChildren>
                            <MenuButton
                                bg="transparent"
                                className="tour-link"
                                width="100%" >
                                <HStack>
                                    <Icon
                                        as={FaQuestionCircle}
                                        color="#82AAAD"
                                        fontSize="xl"
                                        justify="center" />

                                    <Text
                                        align="center"
                                        display={navSize === 'small'
                                            ? 'none'
                                            : 'flex'}
                                        fontSize="m"
                                        pl={5}
                                        pr={10}>
                                        Tutorial
                                    </Text>
                                </HStack>
                            </MenuButton>
                        </Tooltip>
                    </Link>
                </Menu>
            </Flex>

            <JoyRide
                {...tourState}
                callback={callback}
                locale={{
                    last: 'Exit Tutorial'
                }}
                showSkipButton
                styles={{
                    options: {
                        zIndex: 100000
                    },
                    tooltip: {
                        borderRadius: 20
                    },
                    tooltipContainer: {
                        textAlign: 'center'
                    },
                    buttonBack: {
                        marginRight: 10,
                        color: 'blue',
                        outline: 'none'
                    },
                    buttonNext: {
                        backgroundColor: 'blue',
                        borderRadius: 10,
                        outline: 'none'
                    },
                    buttonSkip: {
                        outline: 'none'
                    }
                }}
            />
        </>
    );
};

export default Tour;
