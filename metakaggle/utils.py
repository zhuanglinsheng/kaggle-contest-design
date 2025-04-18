from typing import Literal

type TableName = Literal[
	'Submissions',

	'Kernels',
	'KernelVotes',
	'KernelTags',
	'KernelAcceleratorTypes',
	'KernelLanguages',
	'KernelVersions',
	'KernelVersionDatasetSources',
	'KernelVersionKernelSources',
	'KernelVersionCompetitionSources',
	'KernelVersionModelSources',

	'Models',
	'ModelVersions',
	'ModelTags',
	'ModelVotes',
	'ModelVariations',
	'ModelVariationVersions',

	'Users',
	'UserOrganizations',
	'UserAchievements',
	'UserFollowers',

	'Teams',
	'TeamMemberships',

	'Organizations',

	'Competitions',
	'CompetitionTags',

	'Forums',
	'ForumTopics',
	'ForumMessages',
	'ForumMessageVotes',
	'ForumMessageReactions',

	'Datasets',
	'DatasetVersions',
	'DatasetVotes',
	'DatasetTags',
	'DatasetTasks',
	'DatasetTaskSubmissions',
	'Datasources',

	'Tags',

	'Episodes',
	'EpisodeAgents',
]
