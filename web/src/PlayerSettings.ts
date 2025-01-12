type PlayerSettings = { type: 'human' } | AgentSettings;

export default PlayerSettings;

export type AgentSettings = AlphaZeroSettings;

export type AlphaZeroSettings = {
  type: 'alphaZero';
  numSimulations: number;
  batchSize: number;
};
