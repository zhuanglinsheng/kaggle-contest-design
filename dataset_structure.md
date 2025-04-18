# Stucture of the Dataset `meta-kaggle`

Submissions
- SubmittedUserId --- `Users`
- TeamId --- `Teams`
- SourceKernelVersionId --- `Kernels`

Teams
- CompetitionId --- `Competitions`
- TeamLeaderId --- `Users`
- PublicLeaderboardSubmissionId --- ?
- PrivateLeaderboardSubmissionId --- ?
- WriteUpForumTopicId --- ?

TeamMemberships
- TeamId --- `Teams`
- UserId --- `Users`

Kernels
- AuthorUserId --- `Users`
- CurrentKernelVersionId --- ?
- ForkParentKernelVersionId --- ?
- ForumTopicId --- ?
- FirstKernelVersionId --- ?

Models
- OwnerUserId --- `Users`
- OwnerOrganizationId --- `Organization`
- CurrentModelVersionId --- `ModelVersions`
- ForumId --- `Forum`

ModelVersions
- ModelId --- `Models`
- CreatorUserId --- `Users`

Competitions
- ForumId --- `Forums`
- OrganizationId --- `Organization`
- CompetitionTypeId --- ?

Forums
- ParentForumId --- `Forums`

Datasets
- CreatorUserId --- `Users`
- OwnerUserId --- `Users`
- OwnerOrganizationId --- `Organization`
- CurrentDatasetVersionId --- ?
- CurrentDatasourceVersionId --- ?
- ForumId --- `Forum`

Tags
- ParentTagId --- `Tags`

Episodes
- CompetitionId --- `Competition`

EpisodeAgents
- EpisodeId --- `Episode`
- SubmissionId --- `Submission`

