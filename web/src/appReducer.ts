import PlayerSettings from './PlayerSettings';

export default function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'blackSettingsChange':
      return { ...state, blackSettings: action.settings };
    case 'whiteSettingsChange':
      return { ...state, whiteSettings: action.settings };
    case 'startEndButtonClick':
      if (state.inAnimation) {
        return { ...state, startEndButtonClicked: true };
      }
      return { ...state, inGame: !state.inGame, startEndButtonClicked: false };
    case 'animationStart':
      return { ...state, inAnimation: true };
    case 'animationEnd':
      return {
        ...state,
        inGame: state.startEndButtonClicked ? !state.inGame : state.inGame,
        startEndButtonClicked: false,
        inAnimation: false,
      };
    default:
      console.error('Invalid action type:', action);
      return state;
  }
}

export type AppState = {
  inGame: boolean;
  startEndButtonClicked: boolean;
  inAnimation: boolean;
  blackSettings: PlayerSettings;
  whiteSettings: PlayerSettings;
};

export type AppAction = (
  { type: 'blackSettingsChange', settings: PlayerSettings } |
  { type: 'whiteSettingsChange', settings: PlayerSettings } |
  { type: 'startEndButtonClick' } |
  { type: 'animationStart' } |
  { type: 'animationEnd' }
);
